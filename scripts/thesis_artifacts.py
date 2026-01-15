from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import platform
import re
import subprocess
import sys
from typing import Any, Iterable, Optional


_SAFE_RE = re.compile(r"[^a-z0-9]+")


def _slug(part: str) -> str:
    part = str(part).strip().lower()
    part = _SAFE_RE.sub("_", part)
    part = part.strip("_")
    return part


def _artifact_stem(
    prefix: str,
    topic: str,
    *,
    method: Optional[str] = None,
    dataset: Optional[str] = None,
    split: Optional[str] = None,
    tag: Optional[str] = None,
) -> str:
    parts = [prefix, _slug(topic)]
    if method:
        parts.append(_slug(method))
    if dataset:
        parts.append(_slug(dataset))
    if split:
        parts.append(_slug(split))
    if tag:
        parts.append(_slug(tag))
    return "_".join([p for p in parts if p])


def _ensure_parent(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)


def _git_commit_hash(cwd: Path) -> Optional[str]:
    try:
        out = subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            cwd=str(cwd),
            stderr=subprocess.DEVNULL,
            text=True,
        ).strip()
        return out or None
    except Exception:
        return None


@dataclass
class ThesisArtifactsExporter:
    export_root: Path
    overwrite: bool = True
    export_tag: Optional[str] = None
    exported_files: list[str] = field(default_factory=list)

    def prepare(self) -> None:
        self.export_root.mkdir(parents=True, exist_ok=True)
        for sub in ("figures", "tables", "numbers", "metadata"):
            (self.export_root / sub).mkdir(parents=True, exist_ok=True)

    def _record(self, path: Path) -> None:
        rel = os.path.relpath(path, start=self.export_root)
        rel = rel.replace(os.sep, "/")
        if rel not in self.exported_files:
            self.exported_files.append(rel)

    def figures_dir(self) -> Path:
        return self.export_root / "figures"

    def tables_dir(self) -> Path:
        return self.export_root / "tables"

    def numbers_dir(self) -> Path:
        return self.export_root / "numbers"

    def metadata_dir(self) -> Path:
        return self.export_root / "metadata"

    def savefig(
        self,
        fig: Any,
        topic: str,
        *,
        method: Optional[str] = None,
        dataset: Optional[str] = None,
        split: Optional[str] = None,
        ext: str = "pdf",
        also_png: bool = False,
        dpi: int = 300,
        bbox_inches: str = "tight",
        facecolor: str = "white",
        tag: Optional[str] = None,
    ) -> Path:
        ext = str(ext).lstrip(".").lower()
        stem = _artifact_stem(
            "fig",
            topic,
            method=method,
            dataset=dataset,
            split=split,
            tag=(tag or self.export_tag),
        )
        out = self.figures_dir() / f"{stem}.{ext}"
        _ensure_parent(out)
        if out.exists() and not self.overwrite:
            raise FileExistsError(str(out))
        save_kwargs = {"bbox_inches": bbox_inches, "facecolor": facecolor}
        if ext in {"png", "jpg", "jpeg", "tiff"}:
            save_kwargs["dpi"] = int(dpi)
        fig.savefig(out, **save_kwargs)
        self._record(out)

        if also_png and ext != "png":
            out_png = self.figures_dir() / f"{stem}.png"
            _ensure_parent(out_png)
            if out_png.exists() and not self.overwrite:
                raise FileExistsError(str(out_png))
            fig.savefig(out_png, dpi=int(dpi), bbox_inches=bbox_inches, facecolor=facecolor)
            self._record(out_png)
        return out

    def export_table(
        self,
        df: Any,
        topic: str,
        *,
        method: Optional[str] = None,
        dataset: Optional[str] = None,
        split: Optional[str] = None,
        tag: Optional[str] = None,
        index: bool = False,
        float_format: str = "%.4g",
        caption: Optional[str] = None,
        label: Optional[str] = None,
    ) -> tuple[Path, Path]:
        stem = _artifact_stem(
            "tab",
            topic,
            method=method,
            dataset=dataset,
            split=split,
            tag=(tag or self.export_tag),
        )
        csv_path = self.tables_dir() / f"{stem}.csv"
        tex_path = self.tables_dir() / f"{stem}.tex"
        _ensure_parent(csv_path)
        _ensure_parent(tex_path)
        if (csv_path.exists() or tex_path.exists()) and not self.overwrite:
            raise FileExistsError(f"{csv_path} or {tex_path}")

        df.to_csv(csv_path, index=index)

        latex_kwargs: dict[str, Any] = {
            "index": index,
            "escape": False,
            "float_format": float_format,
        }
        try:
            tex = df.to_latex(**latex_kwargs)
        except TypeError:
            latex_kwargs.pop("float_format", None)
            tex = df.to_latex(**latex_kwargs)

        if caption or label:
            lines = [r"\begin{table}[t]", r"\centering", tex.strip()]
            if caption:
                lines.append(rf"\caption{{{caption}}}")
            if label:
                lines.append(rf"\label{{{label}}}")
            lines.append(r"\end{table}")
            tex = "\n".join(lines) + "\n"
        else:
            tex = tex if tex.endswith("\n") else tex + "\n"

        tex_path.write_text(tex)

        self._record(csv_path)
        self._record(tex_path)
        return tex_path, csv_path

    def export_number(
        self,
        topic: str,
        value: Any,
        *,
        method: Optional[str] = None,
        dataset: Optional[str] = None,
        split: Optional[str] = None,
        tag: Optional[str] = None,
        fmt: str = "{:.6g}",
    ) -> Path:
        stem = _artifact_stem(
            "num",
            topic,
            method=method,
            dataset=dataset,
            split=split,
            tag=(tag or self.export_tag),
        )
        out = self.numbers_dir() / f"{stem}.tex"
        _ensure_parent(out)
        if out.exists() and not self.overwrite:
            raise FileExistsError(str(out))
        try:
            text = fmt.format(value)
        except Exception:
            text = str(value)
        out.write_text(text + "\n")
        self._record(out)
        return out

    def write_manifest(self, *, extra: Optional[dict[str, Any]] = None, project_root: Optional[Path] = None) -> Path:
        out = self.metadata_dir() / "manifest.json"
        _ensure_parent(out)

        project_root = (project_root or self.export_root).resolve()
        out_rel = os.path.relpath(out, start=self.export_root).replace(os.sep, "/")
        exported_files = sorted(set(self.exported_files + [out_rel]))
        payload: dict[str, Any] = {
            "created_at": datetime.now(timezone.utc).isoformat(),
            "git_commit": _git_commit_hash(project_root),
            "export_root": str(self.export_root),
            "overwrite": bool(self.overwrite),
            "export_tag": self.export_tag,
            "exported_files": exported_files,
            "python": sys.version,
            "platform": {
                "node": platform.node(),
                "system": platform.system(),
                "release": platform.release(),
            },
        }
        if extra:
            payload["extra"] = extra

        out.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
        self._record(out)
        return out
