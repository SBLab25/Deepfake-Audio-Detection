import argparse
import csv
import io
import json
import re
import unicodedata
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

import numpy as np
import soundfile as sf
from tqdm import tqdm

try:
    import librosa
except Exception:  # pragma: no cover
    librosa = None

try:
    from datasets import Audio, load_dataset
except Exception as exc:  # pragma: no cover
    raise RuntimeError(
        "Missing dependency: datasets. Install with `pip install datasets`."
    ) from exc


@dataclass
class SourceSpec:
    name: str
    dataset_id: str
    config: Optional[str]
    split: str
    language: str
    label: str
    target_count: int
    filter_mode: str = "none"


def source_audio_dir(root: Path, spec: SourceSpec) -> Path:
    return root / spec.language / spec.label / spec.name


def slugify(text: str) -> str:
    text = unicodedata.normalize("NFKD", text)
    text = text.encode("ascii", "ignore").decode("ascii")
    text = text.lower().strip()
    text = re.sub(r"[^a-z0-9._-]+", "_", text)
    text = re.sub(r"_+", "_", text).strip("_")
    return text or "item"


def safe_get(row: Dict[str, Any], keys: List[str]) -> Optional[Any]:
    for k in keys:
        if k in row and row[k] is not None:
            return row[k]
    return None


def row_key(row: Dict[str, Any]) -> str:
    val = safe_get(
        row,
        [
            "__key__",
            "id",
            "uid",
            "file",
            "filename",
            "path",
            "audio_path",
            "audio_filepath",
            "name",
        ],
    )
    if val is None:
        # Try nested audio dict path/filename
        audio_obj = row.get("audio")
        if isinstance(audio_obj, dict):
            nested = audio_obj.get("path") or audio_obj.get("filename") or audio_obj.get("name")
            if nested:
                return str(nested)
        # Last-resort stable-ish key from text+duration if present
        txt = row.get("text") or row.get("transcription") or row.get("sentence")
        dur = row.get("duration")
        if txt is not None:
            return f"text_{slugify(str(txt)[:80])}_{dur if dur is not None else 'na'}"
        return "unknown"
    return str(val)


def detect_audio_object(row: Dict[str, Any]) -> Tuple[Any, str]:
    preferred = ["audio", "wav", "flac", "speech", "mp3", "waveform"]
    for key in preferred:
        if key in row and row[key] is not None:
            return row[key], key
    for key, val in row.items():
        if isinstance(val, dict) and ("array" in val or "bytes" in val or "path" in val):
            if any(tok in key.lower() for tok in ["audio", "wav", "flac", "speech"]):
                return val, key
    raise ValueError("No audio object found in row.")


def decode_audio(audio_obj: Any, row: Dict[str, Any]) -> Tuple[np.ndarray, int]:
    def _read_bytes(raw: bytes) -> Tuple[np.ndarray, int]:
        with io.BytesIO(raw) as bio:
            try:
                arr, sr = sf.read(bio, dtype="float32", always_2d=False)
                return np.asarray(arr, dtype=np.float32), int(sr)
            except Exception:
                if librosa is None:
                    raise
                bio.seek(0)
                arr, sr = librosa.load(bio, sr=None, mono=False)
                return np.asarray(arr, dtype=np.float32), int(sr)

    # Hugging Face Audio feature (decoded)
    if isinstance(audio_obj, dict):
        if "array" in audio_obj and audio_obj["array"] is not None:
            arr = np.asarray(audio_obj["array"], dtype=np.float32)
            sr = int(
                audio_obj.get("sampling_rate")
                or row.get("sampling_rate")
                or row.get("sr")
                or row.get("sample_rate")
                or 16000
            )
            return arr, sr
        if "bytes" in audio_obj and audio_obj["bytes"] is not None:
            raw = audio_obj["bytes"]
            if isinstance(raw, memoryview):
                raw = raw.tobytes()
            elif isinstance(raw, bytearray):
                raw = bytes(raw)
            if isinstance(raw, bytes):
                return _read_bytes(raw)
        if "path" in audio_obj and audio_obj["path"]:
            p = Path(str(audio_obj["path"]))
            if p.exists():
                arr, sr = sf.read(str(p), dtype="float32", always_2d=False)
                return np.asarray(arr, dtype=np.float32), int(sr)
    # Raw bytes payload (common in webdataset streaming)
    if isinstance(audio_obj, (bytes, bytearray, memoryview)):
        raw = bytes(audio_obj) if not isinstance(audio_obj, bytes) else audio_obj
        return _read_bytes(raw)
    # Some webdataset loaders expose byte list
    if isinstance(audio_obj, list) and audio_obj and isinstance(audio_obj[0], int):
        try:
            raw = bytes(audio_obj)
            return _read_bytes(raw)
        except Exception:
            pass
    # Raw ndarray
    if isinstance(audio_obj, np.ndarray):
        sr = int(row.get("sampling_rate") or row.get("sr") or row.get("sample_rate") or 16000)
        return np.asarray(audio_obj, dtype=np.float32), sr
    # Local file path string
    if isinstance(audio_obj, str):
        p = Path(audio_obj)
        if p.exists():
            arr, sr = sf.read(str(p), dtype="float32", always_2d=False)
            return np.asarray(arr, dtype=np.float32), int(sr)
    raise ValueError("Unsupported audio object format.")


def to_mono_resampled(wav: np.ndarray, sr: int, target_sr: int) -> np.ndarray:
    wav = np.asarray(wav, dtype=np.float32)
    if wav.ndim == 2:
        wav = wav.mean(axis=1)
    if wav.ndim != 1:
        wav = wav.reshape(-1)
    if sr != target_sr:
        if librosa is None:
            raise RuntimeError(
                f"Need librosa to resample {sr} -> {target_sr}. Install librosa."
            )
        wav = librosa.resample(wav, orig_sr=sr, target_sr=target_sr)
    if wav.size == 0:
        raise ValueError("Decoded empty waveform.")
    return wav.astype(np.float32)


def shifty_english_filter(row: Dict[str, Any]) -> bool:
    key = str(row.get("__key__", "")).lower()
    url = str(row.get("__url__", "")).lower()
    blob = f"{key} {url}"
    # Exclude common non-English corpora used in ShiftySpeech
    if any(x in blob for x in ["aishell", "jsut"]):
        return False
    # Keep typical English corpora and domains
    if any(
        x in blob
        for x in [
            "ljspeech",
            "train-clean-360",
            "commonvoice",
            "voxceleb",
            "podcast",
            "youtube",
            "audiobook",
        ]
    ):
        return True
    return False


def row_passes_filter(row: Dict[str, Any], mode: str) -> bool:
    if mode == "none":
        return True
    if mode == "shifty_english":
        return shifty_english_filter(row)
    return True


def infer_speaker_group(spec: SourceSpec, row: Dict[str, Any], key: str) -> str:
    if spec.name == "en_real_librispeech":
        # Typical key: speaker-chapter-utterance
        first = key.split("-")[0]
        if first.isdigit():
            return f"spk_{first}"
    if spec.name == "hi_fake_indicsynth":
        sid = safe_get(row, ["Source Speaker_ID", "source_speaker_id", "speaker_id"])
        if sid is not None:
            return f"spk_{sid}"
    # Generic fallback by normalized key prefix
    return f"grp_{slugify(key)}"


def infer_generator(row: Dict[str, Any], spec: SourceSpec) -> str:
    g = safe_get(
        row,
        [
            "Generative Model",
            "generator",
            "tts_model",
            "model",
            "vocoder",
            "system",
        ],
    )
    if g is None:
        if spec.label == "real":
            return "real_source"
        return spec.name
    return slugify(str(g))


def load_stream(spec: SourceSpec, seed: int, shuffle_buffer: int):
    kwargs: Dict[str, Any] = {"streaming": True, "split": spec.split}
    if spec.config:
        kwargs["name"] = spec.config
    ds = load_dataset(spec.dataset_id, **kwargs)
    # Avoid HF Audio auto-decoding dependency (torchcodec) by keeping raw paths/bytes.
    try:
        if getattr(ds, "features", None):
            for col, feat in ds.features.items():
                if isinstance(feat, Audio):
                    ds = ds.cast_column(col, Audio(decode=False))
    except Exception:
        pass
    ds = ds.shuffle(seed=seed, buffer_size=shuffle_buffer)
    return ds


def iterate_rows(ds: Any, bypass_feature_layer: bool = False) -> Iterable[Dict[str, Any]]:
    if bypass_feature_layer and hasattr(ds, "_ex_iterable"):
        for item in ds._ex_iterable:  # internal API, used intentionally to bypass Audio encoding
            if isinstance(item, tuple) and len(item) == 2:
                _, ex = item
            else:
                ex = item
            if isinstance(ex, dict):
                yield ex
        return
    for ex in ds:
        if isinstance(ex, dict):
            yield ex


def collect_source(
    spec: SourceSpec,
    out_audio_root: Path,
    target_sr: int,
    seed: int,
    shuffle_buffer: int,
    limit_rows: Optional[int],
    max_attempts: int,
    existing_feature_paths: Optional[set] = None,
) -> Tuple[List[Dict[str, Any]], Dict[str, int]]:
    rows: List[Dict[str, Any]] = []
    stats = defaultdict(int)
    taken_ids: Dict[str, int] = defaultdict(int)
    existing_feature_paths = existing_feature_paths or set()

    pbar = tqdm(total=spec.target_count, desc=f"{spec.name}", unit="clip")
    attempt = 0
    bypass_features = spec.name == "en_fake_shifty_tts"
    stop_due_to_limit = False
    total_seen_global = 0
    while len(rows) < spec.target_count and attempt < max_attempts and not stop_due_to_limit:
        attempt += 1
        ds = load_stream(spec, seed=seed + 1009 * attempt, shuffle_buffer=shuffle_buffer)
        it = iter(iterate_rows(ds, bypass_feature_layer=bypass_features))
        while len(rows) < spec.target_count:
            if limit_rows is not None and total_seen_global >= limit_rows:
                stop_due_to_limit = True
                break
            try:
                row = next(it)
            except StopIteration:
                break
            except Exception as exc:
                stats["stream_errors"] += 1
                if stats["stream_errors"] <= 3:
                    print(f"[stream-debug:{spec.name}] {type(exc).__name__}: {exc}")
                break

            total_seen_global += 1
            stats["seen"] += 1
            if not row_passes_filter(row, spec.filter_mode):
                stats["filtered_out"] += 1
                continue

            try:
                audio_obj, _ = detect_audio_object(row)
                wav, sr = decode_audio(audio_obj, row)
                wav = to_mono_resampled(wav, sr=sr, target_sr=target_sr)
            except Exception as exc:
                stats["decode_failed"] += 1
                if stats["decode_failed"] <= 3:
                    preview = {}
                    for k, v in row.items():
                        if isinstance(v, dict):
                            preview[k] = f"dict({','.join(sorted(v.keys()))})"
                        else:
                            preview[k] = type(v).__name__
                    print(
                        f"[decode-debug:{spec.name}] fail#{stats['decode_failed']} "
                        f"error={type(exc).__name__}: {exc} keys={preview}"
                    )
                continue

            key = row_key(row)
            base = slugify(key)
            if base == "unknown":
                base = f"{spec.name}_{stats['seen']:09d}"
            # Ensure key is never unknown for grouping; otherwise an entire source can collapse into one group.
            if key == "unknown":
                key = base
            speaker_group = infer_speaker_group(spec, row, key)
            generator = infer_generator(row, spec)
            dup = taken_ids[base]
            taken_ids[base] += 1
            if dup > 0:
                base = f"{base}_{dup:03d}"

            rel_audio = Path(spec.language) / spec.label / spec.name / f"{base}.wav"
            rel_feat = rel_audio.with_suffix(".npy").as_posix()
            if rel_feat in existing_feature_paths:
                stats["duplicate_skipped"] += 1
                continue

            abs_audio = (out_audio_root / rel_audio).resolve()
            abs_audio.parent.mkdir(parents=True, exist_ok=True)
            try:
                sf.write(str(abs_audio), wav, target_sr)
            except Exception:
                stats["write_failed"] += 1
                continue

            rows.append(
                {
                    "feature_path": rel_feat,
                    "label": spec.label,
                    "language": spec.language,
                    "audio_path": str(abs_audio),
                    "source_dataset": spec.dataset_id,
                    "source_config": spec.config or "",
                    "source_split": spec.split,
                    "source_name": spec.name,
                    "source_key": key,
                    "speaker_group": speaker_group,
                    "generator": generator,
                    "duration_sec": float(len(wav) / max(1, target_sr)),
                }
            )
            existing_feature_paths.add(rel_feat)
            stats["selected"] += 1
            pbar.update(1)
        if len(rows) < spec.target_count and not stop_due_to_limit:
            stats["retries"] += 1
    pbar.close()
    return rows, dict(stats)


def recover_existing_rows_for_source(out_audio_root: Path, spec: SourceSpec) -> List[Dict[str, Any]]:
    src_dir = source_audio_dir(out_audio_root, spec)
    if not src_dir.exists():
        return []
    rows: List[Dict[str, Any]] = []
    for wav_path in sorted(src_dir.rglob("*.wav")):
        rel_audio = wav_path.relative_to(out_audio_root).as_posix()
        stem = wav_path.stem
        key = stem
        speaker_group = infer_speaker_group(spec, {}, key)
        rows.append(
            {
                "feature_path": Path(rel_audio).with_suffix(".npy").as_posix(),
                "label": spec.label,
                "language": spec.language,
                "audio_path": str(wav_path.resolve()),
                "source_dataset": spec.dataset_id,
                "source_config": spec.config or "",
                "source_split": spec.split,
                "source_name": spec.name,
                "source_key": key,
                "speaker_group": speaker_group,
                "generator": infer_generator({}, spec),
                "duration_sec": 0.0,
            }
        )
    return rows


def compute_targets(n: int, train_ratio: float, val_ratio: float) -> Dict[str, int]:
    n_train = int(n * train_ratio)
    n_val = int(n * val_ratio)
    if n_train + n_val >= n:
        n_val = max(0, n - n_train - 1)
    n_test = n - n_train - n_val
    if n_test < 1:
        n_test = 1
        if n_val > 0:
            n_val -= 1
        else:
            n_train = max(1, n_train - 1)
    return {"train": n_train, "val": n_val, "test": n_test}


def grouped_stratified_split(
    rows: List[Dict[str, Any]], train_ratio: float, val_ratio: float, seed: int
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]], List[Dict[str, Any]]]:
    by_stratum: Dict[Tuple[str, str], List[Dict[str, Any]]] = defaultdict(list)
    for r in rows:
        by_stratum[(r["language"], r["label"])].append(r)

    rng = np.random.default_rng(seed)
    train: List[Dict[str, Any]] = []
    val: List[Dict[str, Any]] = []
    test: List[Dict[str, Any]] = []

    for stratum, srows in sorted(by_stratum.items()):
        grouped: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
        for r in srows:
            gid = f"{r['source_name']}|{r['speaker_group']}"
            grouped[gid].append(r)

        groups = list(grouped.values())
        rng.shuffle(groups)
        groups.sort(key=len, reverse=True)

        targets = compute_targets(len(srows), train_ratio, val_ratio)
        counts = {"train": 0, "val": 0, "test": 0}
        alloc = {"train": [], "val": [], "test": []}

        for g in groups:
            size = len(g)
            best_split = None
            best_score = None
            for split in ("train", "val", "test"):
                temp = dict(counts)
                temp[split] += size
                score = sum((temp[s] - targets[s]) ** 2 for s in ("train", "val", "test"))
                if best_score is None or score < best_score:
                    best_score = score
                    best_split = split
            alloc[best_split].extend(g)
            counts[best_split] += size

        train.extend(alloc["train"])
        val.extend(alloc["val"])
        test.extend(alloc["test"])

    rng.shuffle(train)
    rng.shuffle(val)
    rng.shuffle(test)
    return train, val, test


def write_index_csv(path: Path, rows: List[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(
            f, fieldnames=["feature_path", "label", "language", "audio_path"]
        )
        writer.writeheader()
        writer.writerows(
            {
                "feature_path": r["feature_path"],
                "label": r["label"],
                "language": r["language"],
                "audio_path": r["audio_path"],
            }
            for r in rows
        )


def write_manifest(path: Path, rows: List[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "feature_path",
                "label",
                "language",
                "audio_path",
                "source_dataset",
                "source_config",
                "source_split",
                "source_name",
                "source_key",
                "speaker_group",
                "generator",
                "duration_sec",
            ],
        )
        writer.writeheader()
        writer.writerows(rows)


def class_breakdown(rows: List[Dict[str, Any]]) -> Dict[str, int]:
    out: Dict[str, int] = defaultdict(int)
    for r in rows:
        out[f"{r['language']}_{r['label']}"] += 1
    return dict(sorted(out.items()))


def source_breakdown(rows: List[Dict[str, Any]]) -> Dict[str, int]:
    out: Dict[str, int] = defaultdict(int)
    for r in rows:
        out[r["source_name"]] += 1
    return dict(sorted(out.items()))


def overlap_count(a: List[Dict[str, Any]], b: List[Dict[str, Any]], key: str) -> int:
    return len(set(r[key] for r in a) & set(r[key] for r in b))


def build_specs(
    hi_real: int,
    hi_fake: int,
    en_real: int,
    en_fake: int,
    hi_real_source: str,
) -> List[SourceSpec]:
    if hi_real_source == "collabora":
        hi_real_spec = SourceSpec(
            name="hi_real_collabora",
            dataset_id="collabora/hindi-asr-wds",
            config=None,
            split="train",
            language="hi",
            label="real",
            target_count=hi_real,
        )
    else:
        # More reliable for direct audio extraction in this pipeline.
        hi_real_spec = SourceSpec(
            name="hi_real_rahul",
            dataset_id="rahul7star/hindi-speech-dataset",
            config=None,
            split="train",
            language="hi",
            label="real",
            target_count=hi_real,
        )

    return [
        hi_real_spec,
        SourceSpec(
            name="hi_fake_indicsynth",
            dataset_id="vdivyasharma/IndicSynth",
            config="Hindi",
            split="train",
            language="hi",
            label="fake",
            target_count=hi_fake,
        ),
        SourceSpec(
            name="en_real_librispeech",
            dataset_id="openslr/librispeech_asr",
            config="all",
            split="train.clean.360",
            language="en",
            label="real",
            target_count=en_real,
        ),
        SourceSpec(
            name="en_fake_shifty_tts",
            dataset_id="ash56/ShiftySpeech",
            config="tts",
            split="test",
            language="en",
            label="fake",
            target_count=en_fake,
            filter_mode="shifty_english",
        ),
    ]


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Build a bilingual Hindi+English deepfake dataset (150k-200k) from Hugging Face "
            "with local audio files and split CSVs."
        )
    )
    parser.add_argument("--project-root", type=str, default=".")
    parser.add_argument("--out-prefix", type=str, default="Composite180k")
    parser.add_argument("--total", type=int, default=180000)
    parser.add_argument("--hi-real", type=int, default=45000)
    parser.add_argument("--hi-fake", type=int, default=45000)
    parser.add_argument("--en-real", type=int, default=45000)
    parser.add_argument("--en-fake", type=int, default=45000)
    parser.add_argument("--train-ratio", type=float, default=0.8)
    parser.add_argument("--val-ratio", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--hi-real-source",
        type=str,
        choices=["rahul", "collabora"],
        default="rahul",
        help="Hindi real source dataset. 'rahul' is default because it is easier to decode end-to-end.",
    )
    parser.add_argument("--target-sr", type=int, default=16000)
    parser.add_argument("--shuffle-buffer", type=int, default=20000)
    parser.add_argument("--source-max-attempts", type=int, default=5)
    parser.add_argument(
        "--limit-rows-per-source",
        type=int,
        default=None,
        help="Optional hard cap for debugging dry runs.",
    )
    parser.add_argument(
        "--allow-shortfall",
        action="store_true",
        help="Do not fail if a source cannot reach requested count.",
    )
    parser.add_argument(
        "--reuse-existing-audio",
        action="store_true",
        default=True,
        help="Reuse already downloaded source audio under <out-prefix>_raw_audio (default: true).",
    )
    args = parser.parse_args()

    if not (150000 <= args.total <= 200000):
        raise ValueError("--total must be between 150000 and 200000.")
    if args.hi_real + args.hi_fake + args.en_real + args.en_fake != args.total:
        raise ValueError("Quadrant counts must sum exactly to --total.")
    if args.train_ratio <= 0 or args.val_ratio <= 0 or args.train_ratio + args.val_ratio >= 1:
        raise ValueError("train_ratio and val_ratio must be > 0 and sum to < 1.")

    root = Path(args.project_root).expanduser().resolve()
    out_prefix = slugify(args.out_prefix)
    raw_root = root / f"{out_prefix}_raw_audio"
    manifest_dir = root / f"{out_prefix}_manifest"
    manifest_dir.mkdir(parents=True, exist_ok=True)

    specs = build_specs(
        hi_real=args.hi_real,
        hi_fake=args.hi_fake,
        en_real=args.en_real,
        en_fake=args.en_fake,
        hi_real_source=args.hi_real_source,
    )

    all_rows: List[Dict[str, Any]] = []
    source_stats: Dict[str, Dict[str, int]] = {}
    for spec in specs:
        print(f"\nCollecting source: {spec.name}")
        existing_rows: List[Dict[str, Any]] = []
        if args.reuse_existing_audio:
            existing_rows = recover_existing_rows_for_source(raw_root, spec)
            if existing_rows:
                print(f"  Reused existing {spec.name} rows: {len(existing_rows)}")

        needed = max(0, spec.target_count - len(existing_rows))
        new_rows: List[Dict[str, Any]] = []
        stats: Dict[str, int] = {"reused": len(existing_rows)}
        if needed > 0:
            spec_needed = SourceSpec(
                name=spec.name,
                dataset_id=spec.dataset_id,
                config=spec.config,
                split=spec.split,
                language=spec.language,
                label=spec.label,
                target_count=needed,
                filter_mode=spec.filter_mode,
            )
            new_rows, new_stats = collect_source(
                spec=spec_needed,
                out_audio_root=raw_root,
                target_sr=args.target_sr,
                seed=args.seed,
                shuffle_buffer=args.shuffle_buffer,
                limit_rows=args.limit_rows_per_source,
                max_attempts=args.source_max_attempts,
                existing_feature_paths=set(r["feature_path"] for r in existing_rows),
            )
            stats.update(new_stats)
        rows = existing_rows + new_rows

        all_rows.extend(rows)
        source_stats[spec.name] = stats
        selected_total = len(rows)
        selected_new = stats.get("selected", 0)
        print(
            f"  selected_total={selected_total}/{spec.target_count} "
            f"(new={selected_new}, reused={stats.get('reused', 0)}) "
            f"seen={stats.get('seen', 0)} "
            f"decode_failed={stats.get('decode_failed', 0)} "
            f"filtered_out={stats.get('filtered_out', 0)}"
        )
        if selected_total < spec.target_count and not args.allow_shortfall:
            raise RuntimeError(
                f"Source {spec.name} shortfall: got {selected_total}, required {spec.target_count}. "
                "Use --allow-shortfall to continue."
            )

    if len(all_rows) < 1:
        raise RuntimeError("No samples were collected.")

    full_manifest = manifest_dir / "full_manifest.csv"
    write_manifest(full_manifest, all_rows)

    train_rows, val_rows, test_rows = grouped_stratified_split(
        rows=all_rows,
        train_ratio=args.train_ratio,
        val_ratio=args.val_ratio,
        seed=args.seed,
    )

    train_csv = root / f"{out_prefix}_train" / f"{out_prefix}_train" / "balanced_index.csv"
    val_csv = root / f"{out_prefix}_val" / f"{out_prefix}_val" / "balanced_index.csv"
    test_csv = root / f"{out_prefix}_test" / f"{out_prefix}_test" / "balanced_index.csv"
    write_index_csv(train_csv, train_rows)
    write_index_csv(val_csv, val_rows)
    write_index_csv(test_csv, test_rows)

    summary = {
        "out_prefix": out_prefix,
        "raw_audio_root": str(raw_root),
        "full_manifest": str(full_manifest),
        "total_collected": len(all_rows),
        "source_stats": source_stats,
        "splits": {
            "train_csv": str(train_csv),
            "val_csv": str(val_csv),
            "test_csv": str(test_csv),
            "train_rows": len(train_rows),
            "val_rows": len(val_rows),
            "test_rows": len(test_rows),
            "train_breakdown": class_breakdown(train_rows),
            "val_breakdown": class_breakdown(val_rows),
            "test_breakdown": class_breakdown(test_rows),
            "train_sources": source_breakdown(train_rows),
            "val_sources": source_breakdown(val_rows),
            "test_sources": source_breakdown(test_rows),
        },
        "overlap_checks": {
            "audio_path_train_val": overlap_count(train_rows, val_rows, "audio_path"),
            "audio_path_train_test": overlap_count(train_rows, test_rows, "audio_path"),
            "audio_path_val_test": overlap_count(val_rows, test_rows, "audio_path"),
            "feature_path_train_val": overlap_count(train_rows, val_rows, "feature_path"),
            "feature_path_train_test": overlap_count(train_rows, test_rows, "feature_path"),
            "feature_path_val_test": overlap_count(val_rows, test_rows, "feature_path"),
        },
        "seed": args.seed,
        "train_ratio": args.train_ratio,
        "val_ratio": args.val_ratio,
        "test_ratio": round(1.0 - args.train_ratio - args.val_ratio, 6),
    }
    summary_path = root / f"{out_prefix}_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    print("\nBuild complete.")
    print(f"Total collected: {len(all_rows)}")
    print(f"Train/Val/Test: {len(train_rows)}/{len(val_rows)}/{len(test_rows)}")
    print("Overlap checks:", summary["overlap_checks"])
    print(f"Summary: {summary_path}")


if __name__ == "__main__":
    main()
