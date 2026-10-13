#!/usr/bin/env python3
"""Human-in-the-loop Meta layer for the flat Phishing Detection capstone folder.

The Email and URL predictors run independently. URLs extracted from email are
scanned automatically; URLs can also be supplied without an email. Visual is
optional and is loaded only after explicit human approval for supplied local
screenshots. This program never fetches URLs or automatically blocks anything.
"""
from __future__ import annotations

import argparse
import contextlib
import email
import html
import importlib.machinery
import importlib.util
import inspect
import json
import re
import sys
from datetime import datetime
from email import policy
from email.parser import BytesParser
from pathlib import Path
from typing import Any

BASE_DIR = Path(__file__).resolve().parent
# Prefer the user's current flat layout, but retain split-folder compatibility.
EMAIL_DIR_CANDIDATES = [BASE_DIR, BASE_DIR / "Email_layer_split", BASE_DIR / "email_layer_split"]
URL_DIR_CANDIDATES = [BASE_DIR, BASE_DIR / "URL_layer_split", BASE_DIR / "url_layer_split"]
VISUAL_DIR_CANDIDATES = [BASE_DIR, BASE_DIR / "visual_layer_split", BASE_DIR / "Visual_layer_split"]
EMAIL_SCRIPT_NAMES = [
    "Email_Inference.py", "Email_layer_inference.py", "Email_layer_interference.py",
    "email_inference.py", "inference.py",
]
URL_SCRIPT_NAMES = [
    "URL_layer_inference.py", "URL_Inference.py", "URL_Inference",
    "url_inference.py", "url_layer_inference.py",
]
VISUAL_SCRIPT_NAMES = [
    "visual_layer_inference.py", "Visual_Inference.py", "visual_inference.py",
]
URL_RE = re.compile(r"(?i)(?:https?://|www\.)[^\s<>\"']+")
URL_TRAILING_PUNCTUATION = ")]}>.,;!?\"'"
IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".webp", ".bmp", ".tif", ".tiff"}


def _find_existing_script(directory: Path, names: list[str]) -> Path | None:
    for name in names:
        candidate = directory / name
        if candidate.is_file():
            return candidate.resolve()
    return None


def choose_dir(explicit: str | None, candidates: list[Path], script_names: list[str]) -> Path | None:
    if explicit:
        return Path(explicit).expanduser().resolve()
    for candidate in candidates:
        if candidate.is_dir() and _find_existing_script(candidate, script_names):
            return candidate.resolve()
    return None


def import_from_path(module_name: str, script_path: Path):
    """Import Python source by path, including an extensionless .py source file."""
    script_path = script_path.expanduser().resolve()
    if not script_path.is_file():
        raise FileNotFoundError(f"Inference/module file not found: {script_path}")
    parent = str(script_path.parent)
    if parent in sys.path:
        sys.path.remove(parent)
    sys.path.insert(0, parent)

    # SourceFileLoader works for ordinary .py and extensionless Python source.
    loader = importlib.machinery.SourceFileLoader(module_name, str(script_path))
    spec = importlib.util.spec_from_loader(module_name, loader)
    if spec is None or spec.loader is None:
        raise ImportError(f"Could not load Python source from {script_path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


def _install_sibling_alias(directory: Path, candidate_names: list[str], aliases: list[str]) -> Path | None:
    """Load a sibling module once, then expose common legacy/case aliases."""
    path = next((directory / name for name in candidate_names if (directory / name).is_file()), None)
    if path is None:
        return None
    # Reuse the module if one of its aliases already points at this same file.
    resolved = path.resolve()
    module = None
    for alias in aliases:
        existing = sys.modules.get(alias)
        if existing is not None and Path(getattr(existing, "__file__", "")).resolve() == resolved:
            module = existing
            break
    if module is None:
        module = import_from_path(aliases[0], resolved)
    for alias in aliases:
        sys.modules[alias] = module
    return resolved


def import_email_features_alias(email_dir: Path) -> None:
    """Make the project's Email_Features.py importable as legacy `features`."""
    _install_sibling_alias(
        email_dir,
        ["Email_Features.py", "Email_layer_features.py", "features.py", "email_features.py"],
        ["features", "Email_Features", "Email_layer_features"],
    )


def prepare_url_import_aliases(url_dir: Path) -> None:
    _install_sibling_alias(
        url_dir,
        ["URL_Features.py", "URL_layer_features.py", "url_features.py"],
        ["URL_layer_features", "URL_Features", "url_features"],
    )


def prepare_visual_import_aliases(visual_dir: Path) -> None:
    _install_sibling_alias(
        visual_dir,
        ["Visual_Model.py", "visual_layer_model.py", "visual_model.py"],
        ["visual_layer_model", "Visual_Model", "visual_model"],
    )
    _install_sibling_alias(
        visual_dir,
        ["Visual_Explainability.py", "visual_layer_explainability.py", "visual_explainability.py"],
        ["visual_layer_explainability", "Visual_Explainability", "visual_explainability"],
    )


def discover_email_script(email_dir: Path) -> Path:
    script = _find_existing_script(email_dir, EMAIL_SCRIPT_NAMES)
    if script:
        return script
    raise FileNotFoundError(
        f"Cannot find Email inference in {email_dir}. Expected one of: {', '.join(EMAIL_SCRIPT_NAMES)}"
    )


def discover_url_script(url_dir: Path) -> Path:
    script = _find_existing_script(url_dir, URL_SCRIPT_NAMES)
    if script:
        return script
    raise FileNotFoundError(
        f"Cannot find URL inference in {url_dir}. Expected one of: {', '.join(URL_SCRIPT_NAMES)}"
    )


def discover_visual_script(visual_dir: Path) -> Path:
    script = _find_existing_script(visual_dir, VISUAL_SCRIPT_NAMES)
    if script:
        return script
    raise FileNotFoundError(
        f"Cannot find Visual inference in {visual_dir}. Expected one of: {', '.join(VISUAL_SCRIPT_NAMES)}"
    )


def _script_model_references(script_path: Path) -> list[str]:
    """Extract literal model artifact names from common MODEL_PATH/DEFAULT_MODEL assignments."""
    try:
        source = script_path.read_text(encoding="utf-8", errors="replace")
    except OSError:
        return []
    references: list[str] = []
    assignment = re.compile(
        r"(?im)^\s*(?:MODEL_PATH|DEFAULT_MODEL|DEFAULT_MODEL_PATH|MODEL_FILE|CHECKPOINT_PATH)\s*=([^\n]+)"
    )
    literal = re.compile(r"['\"]([^'\"]+\.(?:joblib|pkl|pt|pth|ckpt))['\"]", re.I)
    for line in assignment.findall(source):
        references.extend(literal.findall(line))
    return references


def resolve_model_path(explicit: str | None, script_path: Path | None, search_dirs: list[Path],
                       default: Path, extensions: tuple[str, ...] = (".joblib",)) -> Path:
    if explicit:
        return Path(explicit).expanduser().resolve()
    candidates: list[Path] = []
    if script_path:
        for reference in _script_model_references(script_path):
            referenced = Path(reference)
            if referenced.is_absolute():
                candidates.append(referenced)
            else:
                candidates.append(script_path.parent / referenced)
                for parent in search_dirs:
                    candidates.append(parent / referenced)
    for parent in search_dirs:
        for name in ("phishing_model.joblib", "email_model_tuned.joblib", "url_model.joblib",
                     "url_model_v3.joblib", "url_model_tuned.joblib", "phish360_efficientnet_b0_best.pt"):
            candidates.append(parent / name)
        # Support split exports as a fallback even when running from the flat root.
        candidates.extend([
            parent / "models" / "url_model.joblib",
            parent / "artifacts" / "visual" / "phish360_efficientnet_b0_best.pt",
        ])
    for candidate in candidates:
        if candidate.is_file() and (not extensions or candidate.suffix.lower() in extensions):
            return candidate.resolve()
    return default.resolve()


class _URLFunctionAdapter:
    """Small compatibility adapter for URL inference modules exposing functions, not a class."""
    def __init__(self, module, model_path: Path):
        self.module = module
        self.model_path = model_path
        loader = getattr(module, "load_predictor", None) or getattr(module, "load_model", None)
        self.predictor = None
        if callable(loader):
            try:
                self.predictor = loader(str(model_path))
            except TypeError:
                self.predictor = loader(model_path=str(model_path))

    def predict_batch(self, urls: list[str]):
        target = self.predictor or self.module
        fn = getattr(target, "predict_batch", None) or getattr(target, "predict_urls", None)
        if callable(fn):
            try:
                return fn(urls)
            except TypeError:
                return fn(urls=urls)
        fn = getattr(target, "predict_url", None) or getattr(target, "predict", None)
        if callable(fn):
            return [fn(url) for url in urls]
        raise AttributeError(
            f"{self.module.__file__} does not expose URLPhishingPredictor or a supported predict_batch/predict_url function"
        )


class LayerRunners:
    """Load layer models once; an unavailable layer does not stop other layers."""
    def __init__(self, args: argparse.Namespace):
        self.email = None
        self.email_error = None
        self.url = None
        self.url_error = None
        self.visual_module = None
        self.visual_model = None
        self.visual_device = None
        self.visual_model_module = None
        self.visual_error = None
        self._email_load_attempted = False
        self._url_load_attempted = False
        self._visual_load_attempted = False
        self._email_script_override = args.email_script
        self._email_model_override = args.email_model
        self._url_model_override = args.url_model
        self._visual_model_override = args.visual_model

        self.email_dir = self._resolve_email_dir(args)
        self.url_dir = choose_dir(args.url_dir, URL_DIR_CANDIDATES, URL_SCRIPT_NAMES)
        self.visual_dir = choose_dir(args.visual_dir, VISUAL_DIR_CANDIDATES, VISUAL_SCRIPT_NAMES)
        # Each layer is lazy-loaded only when its input is actually present.

    @staticmethod
    def _resolve_email_dir(args: argparse.Namespace) -> Path | None:
        if args.email_script:
            return Path(args.email_script).expanduser().resolve().parent
        if args.email_layer_dir:
            return Path(args.email_layer_dir).expanduser().resolve()
        for directory in EMAIL_DIR_CANDIDATES:
            if directory.is_dir() and _find_existing_script(directory, EMAIL_SCRIPT_NAMES):
                return directory.resolve()
        return None

    def _load_email(self, explicit_script: str | None, explicit_model: str | None) -> None:
        if self._email_load_attempted:
            return
        self._email_load_attempted = True
        try:
            email_dir = Path(explicit_script).expanduser().resolve().parent if explicit_script else self.email_dir
            if email_dir is None:
                raise FileNotFoundError("Email inference folder was not found; pass --email-layer-dir or --email-script.")
            script_path = Path(explicit_script).expanduser().resolve() if explicit_script else discover_email_script(email_dir)
            import_email_features_alias(email_dir)
            module = import_from_path("_meta_email_inference", script_path)
            predictor_class = getattr(module, "PhishingPredictor", None)
            if predictor_class is None:
                raise AttributeError(f"{script_path.name} does not expose PhishingPredictor.")
            model_path = resolve_model_path(
                explicit_model, script_path, [email_dir, BASE_DIR],
                email_dir / "phishing_model.joblib", extensions=(".joblib", ".pkl"),
            )
            self.email = predictor_class(str(model_path))
        except Exception as exc:
            self.email_error = f"{type(exc).__name__}: {exc}"

    def _load_url(self, explicit_model: str | None) -> None:
        if self._url_load_attempted:
            return
        self._url_load_attempted = True
        try:
            if self.url_dir is None:
                raise FileNotFoundError("URL inference was not found in the flat project root or URL split folder; pass --url-dir.")
            script_path = discover_url_script(self.url_dir)
            prepare_url_import_aliases(self.url_dir)
            module = import_from_path("_meta_url_inference", script_path)
            model_path = resolve_model_path(
                explicit_model, script_path, [self.url_dir, BASE_DIR],
                self.url_dir / "url_model.joblib", extensions=(".joblib", ".pkl"),
            )
            predictor_class = next((getattr(module, name, None) for name in
                                    ("URLPhishingPredictor", "URLPredictor", "URLClassifier")
                                    if callable(getattr(module, name, None))), None)
            if predictor_class is not None:
                self.url = predictor_class(str(model_path))
            else:
                self.url = _URLFunctionAdapter(module, model_path)
                # Validate there is a usable predict API before treating adapter as loaded.
                if not (callable(getattr(self.url.predictor or module, "predict_batch", None)) or
                        callable(getattr(self.url.predictor or module, "predict_urls", None)) or
                        callable(getattr(self.url.predictor or module, "predict_url", None)) or
                        callable(getattr(self.url.predictor or module, "predict", None))):
                    raise AttributeError(f"{script_path.name} has no recognised URL prediction API.")
        except Exception as exc:
            self.url_error = f"{type(exc).__name__}: {exc}"

    def _load_visual(self) -> None:
        """Load Visual code and checkpoint only after the operator approves analysis."""
        if self._visual_load_attempted:
            return
        self._visual_load_attempted = True
        try:
            if self.visual_dir is None:
                raise FileNotFoundError("Visual inference was not found in the flat project root or Visual split folder; pass --visual-dir.")
            script_path = discover_visual_script(self.visual_dir)
            prepare_visual_import_aliases(self.visual_dir)
            module = import_from_path("_meta_visual_inference", script_path)
            checkpoint = resolve_model_path(
                self._visual_model_override, script_path, [self.visual_dir, BASE_DIR],
                self.visual_dir / "phish360_efficientnet_b0_best.pt", extensions=(".pt", ".pth", ".ckpt"),
            )
            if not checkpoint.is_file():
                raise FileNotFoundError(f"Visual checkpoint not found: {checkpoint}")

            loader = getattr(module, "load_predictor", None) or getattr(module, "load_model", None)
            model, device = None, None
            if callable(loader):
                try:
                    signature = inspect.signature(loader)
                    parameters = signature.parameters
                except (TypeError, ValueError):
                    parameters = {}
                path_parameter = next((name for name in parameters if name.lower() in {
                    "checkpoint_path", "model_path", "checkpoint", "weights_path", "model_file", "path"
                }), None)
                required_parameters = [p for p in parameters.values()
                                       if p.default is inspect.Parameter.empty and
                                       p.kind in (inspect.Parameter.POSITIONAL_ONLY,
                                                  inspect.Parameter.POSITIONAL_OR_KEYWORD,
                                                  inspect.Parameter.KEYWORD_ONLY)]
                if path_parameter:
                    loaded = loader(**{path_parameter: str(checkpoint)})
                elif not required_parameters:
                    loaded = loader()
                elif required_parameters and required_parameters[0].name.lower() in {"device", "torch_device"}:
                    import torch
                    runtime_device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
                    loaded = loader(**{required_parameters[0].name: runtime_device})
                else:
                    loaded = loader(str(checkpoint))
                if isinstance(loaded, tuple) and len(loaded) >= 2:
                    model, device = loaded[0], loaded[1]
                elif loaded is not None:
                    model = loaded
                    device = getattr(module, "DEVICE", getattr(module, "device", None))
                else:
                    model = next((getattr(module, name, None) for name in
                                  ("model", "MODEL", "predictor", "MODEL_INSTANCE")
                                  if getattr(module, name, None) is not None), None)
                    device = getattr(module, "DEVICE", getattr(module, "device", None))
            else:
                # Compatibility fallback for the Visual_Model.py + checkpoint layout.
                model_module = sys.modules.get("visual_layer_model")
                build_model = getattr(model_module, "build_model", None) if model_module else None
                if not callable(build_model):
                    raise AttributeError(f"{script_path.name} does not expose load_predictor/load_model, and Visual_Model.py has no build_model().")
                import torch
                device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
                loaded_checkpoint = torch.load(str(checkpoint), map_location=device, weights_only=False)
                try:
                    model = build_model(device=device, pretrained=False)
                except TypeError:
                    try:
                        model = build_model(device=device)
                    except TypeError:
                        model = build_model()
                        if hasattr(model, "to"):
                            model = model.to(device)
                state_dict = loaded_checkpoint.get("state_dict", loaded_checkpoint) if isinstance(loaded_checkpoint, dict) else loaded_checkpoint
                model.load_state_dict(state_dict)
                model.eval()

            if not callable(getattr(module, "predict_image", None)):
                raise AttributeError(f"{script_path.name} does not expose predict_image().")
            self.visual_module = module
            self.visual_model_module = sys.modules.get("visual_layer_model")
            self.visual_model = model
            self.visual_device = device
        except Exception as exc:
            self.visual_error = f"{type(exc).__name__}: {exc}"

    def predict_email(self, subject: str, body: str) -> dict[str, Any]:
        if not self._email_load_attempted:
            self._load_email(self._email_script_override, self._email_model_override)
        if self.email is None:
            return {"status": "unavailable", "error": self.email_error or "Email model unavailable"}
        try:
            result = self.email.predict(subject, body)
            return {"status": "ok", "result": _json_safe(result)}
        except Exception as exc:
            return {"status": "error", "error": f"{type(exc).__name__}: {exc}"}

    def predict_urls(self, urls: list[str]) -> dict[str, Any]:
        if not urls:
            return {"status": "skipped", "reason": "SKIPPED_NO_URL", "message": "No URL was extracted or supplied manually."}
        if not self._url_load_attempted:
            self._load_url(self._url_model_override)
        if self.url is None:
            return {"status": "unavailable", "error": self.url_error or "URL model unavailable", "url_count": len(urls)}
        try:
            results = self.url.predict_batch(urls)
            if isinstance(results, dict):
                results = [results]
            elif not isinstance(results, (list, tuple)):
                results = list(results) if results is not None else []
            return {"status": "ok", "url_count": len(urls), "results": _json_safe(results)}
        except Exception as exc:
            return {"status": "error", "error": f"{type(exc).__name__}: {exc}", "url_count": len(urls)}

    def _call_visual_predict(self, image_path: Path, gradcam: bool, gradcam_path: Path | None):
        fn = self.visual_module.predict_image
        try:
            parameters = inspect.signature(fn).parameters
            names = list(parameters)
        except (TypeError, ValueError):
            names = []
        first = names[0].lower() if names else "model"
        kwargs: dict[str, Any] = {}
        if "gradcam" in names:
            kwargs["gradcam"] = gradcam
        elif "show_gradcam" in names:
            kwargs["show_gradcam"] = gradcam
        if "gradcam_out" in names:
            kwargs["gradcam_out"] = str(gradcam_path) if gradcam_path else None
        elif "output_path" in names and gradcam_path:
            kwargs["output_path"] = str(gradcam_path)

        # New split API: predict_image(model, image_path, device, ...).
        if first in {"model", "predictor", "net", "classifier"}:
            args = [self.visual_model, str(image_path)]
            if len(names) >= 3:
                args.append(self.visual_device)
            return fn(*args, **kwargs)
        # Legacy/helper API with module-level model: predict_image(image_path, ...).
        if first in {"image_path", "path", "img_path", "image", "filepath", "file_path"}:
            for name in ("model", "MODEL", "predictor", "MODEL_INSTANCE"):
                if hasattr(self.visual_module, name):
                    setattr(self.visual_module, name, self.visual_model)
            for name in ("DEVICE", "device"):
                if hasattr(self.visual_module, name) and self.visual_device is not None:
                    setattr(self.visual_module, name, self.visual_device)
            return fn(str(image_path), **kwargs)
        # Last-resort convention matching the documented split API.
        return fn(self.visual_model, str(image_path), self.visual_device, **kwargs)

    @staticmethod
    def _normalise_visual_result(result: Any, image_path: Path) -> dict[str, Any]:
        if isinstance(result, dict):
            result = _json_safe(result)
            result.setdefault("image_path", str(image_path))
            if "label" in result:
                result["label"] = str(result["label"]).replace("🚨", "").replace("✅", "").strip()
            return result
        # Legacy Visual_test-style return: (class_index, probability vector).
        if isinstance(result, (tuple, list)) and len(result) >= 2:
            pred, probabilities = result[0], result[1]
            try:
                probs = probabilities.tolist() if hasattr(probabilities, "tolist") else list(probabilities)
                probs = [float(x) for x in probs]
                label = "phishing" if int(pred) == 1 else "legitimate"
                return {"image_path": str(image_path), "label": label,
                        "probability_phishing": probs[1] if len(probs) > 1 else None,
                        "probability_legitimate": probs[0] if probs else None}
            except Exception:
                pass
        if isinstance(result, (str, int, bool)):
            label = ("phishing" if result == 1 else "legitimate") if isinstance(result, (int, bool)) else str(result)
            return {"image_path": str(image_path), "label": label}
        raise TypeError(f"Unsupported result from Visual predict_image(): {type(result).__name__}")

    def predict_images(self, image_paths: list[Path], gradcam: bool, evidence_dir: Path) -> dict[str, Any]:
        if not image_paths:
            return {"status": "skipped", "reason": "SKIPPED_NO_IMAGE", "message": "No screenshot was supplied or matched."}
        if self.visual_module is None or self.visual_model is None:
            self._load_visual()
        if self.visual_module is None or self.visual_model is None:
            return {"status": "unavailable", "error": self.visual_error or "Visual model unavailable", "image_count": len(image_paths)}
        output = []
        for image_path in image_paths:
            try:
                gradcam_path = None
                if gradcam:
                    evidence_dir.mkdir(parents=True, exist_ok=True)
                    gradcam_path = evidence_dir / f"{image_path.stem}_gradcam.png"
                # Suppress per-layer CLI prints; Meta will issue one unified report.
                import io
                with contextlib.redirect_stdout(io.StringIO()):
                    raw_result = self._call_visual_predict(image_path, gradcam, gradcam_path)
                item = self._normalise_visual_result(raw_result, image_path)
                if gradcam_path and gradcam_path.is_file():
                    item["gradcam_path"] = str(gradcam_path.resolve())
                elif gradcam_path:
                    item["gradcam_requested_path"] = str(gradcam_path.resolve())
                output.append(item)
            except Exception as exc:
                output.append({"image_path": str(image_path), "error": f"{type(exc).__name__}: {exc}"})
        successful = [x for x in output if "label" in x]
        failures = [x for x in output if "error" in x]
        status = "ok" if successful and not failures else "partial" if successful else "error"
        return {"status": status, "image_count": len(image_paths), "results": output}

def _json_safe(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(k): _json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(v) for v in value]
    if isinstance(value, Path):
        return str(value)
    if hasattr(value, "item"):
        try:
            return value.item()
        except Exception:
            pass
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    return str(value)


def extract_urls(text: str) -> list[str]:
    seen = set()
    urls: list[str] = []
    for match in URL_RE.findall(text or ""):
        cleaned = match.rstrip(URL_TRAILING_PUNCTUATION)
        if cleaned and cleaned.lower() not in seen:
            seen.add(cleaned.lower())
            urls.append(cleaned)
    return urls


def _part_to_text(part) -> str:
    try:
        payload = part.get_content()
        return payload if isinstance(payload, str) else str(payload)
    except Exception:
        try:
            raw = part.get_payload(decode=True) or b""
            charset = part.get_content_charset() or "utf-8"
            return raw.decode(charset, errors="replace")
        except Exception:
            return ""


def load_email_file(path: Path) -> tuple[str, str]:
    """Return (subject, body); supports .eml and plain .txt files."""
    if path.suffix.lower() == ".eml":
        with path.open("rb") as f:
            message = BytesParser(policy=policy.default).parse(f)
        subject = str(message.get("subject", "") or "")
        plain_parts, html_parts = [], []
        parts = message.walk() if message.is_multipart() else [message]
        for part in parts:
            if part.get_content_maintype() != "text" or part.get_content_disposition() == "attachment":
                continue
            text = _part_to_text(part)
            if part.get_content_subtype() == "plain":
                plain_parts.append(text)
            elif part.get_content_subtype() == "html":
                html_parts.append(text)
        body = "\n".join(plain_parts if plain_parts else html_parts)
        return subject, body
    return "", path.read_text(encoding="utf-8-sig", errors="replace")


def collect_inputs(args: argparse.Namespace) -> list[dict[str, Any]]:
    """Load email records. URL-only and Visual-only inputs are created in main()."""
    if args.email_dir:
        root = Path(args.email_dir).expanduser().resolve()
        if not root.is_dir():
            raise NotADirectoryError(f"Email input directory not found: {root}")
        files = sorted(p for p in root.rglob("*") if p.is_file() and p.suffix.lower() in {".eml", ".txt"})
        if not files:
            raise FileNotFoundError(f"No .eml or .txt files found under {root}")
        inputs = []
        for path in files:
            try:
                subject, body = load_email_file(path)
                inputs.append({"source": str(path), "name": path.stem, "subject": subject, "body": body, "email_present": True})
            except Exception as exc:
                inputs.append({"source": str(path), "name": path.stem, "subject": "", "body": "", "email_present": True,
                               "input_error": f"{type(exc).__name__}: {exc}"})
        return inputs
    if args.email_file:
        path = Path(args.email_file).expanduser().resolve()
        subject, body = load_email_file(path)
        return [{"source": str(path), "name": path.stem, "subject": subject, "body": body, "email_present": True}]
    if args.subject is not None or bool(args.body):
        return [{"source": "direct_input", "name": "direct_input", "subject": args.subject or "", "body": args.body or "",
                 "email_present": True}]
    return []


def pair_screenshots(record: dict[str, Any], args: argparse.Namespace, record_count: int) -> list[Path]:
    if args.images:
        if record_count == 1:
            paths = [Path(p).expanduser().resolve() for p in args.images]
            return [p for p in paths if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS]
    if args.screenshots_dir:
        root = Path(args.screenshots_dir).expanduser().resolve()
        if not root.is_dir():
            return []
        stem = record.get("name", "")
        candidates = list(root.glob(f"{stem}.*")) + list(root.glob(f"{stem}__*"))
        seen, result = set(), []
        for path in sorted(candidates):
            if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS and path.resolve() not in seen:
                seen.add(path.resolve())
                result.append(path.resolve())
        return result
    return []


def unique_urls(urls: list[str]) -> list[str]:
    seen: set[str] = set()
    result: list[str] = []
    for url in urls:
        candidate = str(url).strip().rstrip(URL_TRAILING_PUNCTUATION)
        if candidate and candidate.lower() not in seen:
            seen.add(candidate.lower())
            result.append(candidate)
    return result


def is_phishing_label(value: Any) -> bool:
    return str(value or "").strip().lower() in {"phishing", "phish", "malicious", "fraudulent"}


def assess_layers(email_result: dict[str, Any], url_result: dict[str, Any],
                  visual_result: dict[str, Any] | None = None) -> dict[str, Any]:
    """Create a transparent evidence-based triage; do not merge uncalibrated probabilities."""
    visual_result = visual_result or {"status": "skipped", "reason": "NOT_INCLUDED_IN_ASSESSMENT"}
    email_prediction = email_result.get("result") if email_result.get("status") == "ok" else None
    url_predictions = url_result.get("results", []) if url_result.get("status") == "ok" else []
    visual_predictions = visual_result.get("results", []) if visual_result.get("status") in {"ok", "partial"} else []
    url_predictions = [x for x in url_predictions if isinstance(x, dict)]
    visual_predictions = [x for x in visual_predictions if isinstance(x, dict) and "label" in x]

    email_flagged = isinstance(email_prediction, dict) and is_phishing_label(email_prediction.get("label"))
    flagged_urls = [x for x in url_predictions if is_phishing_label(x.get("label"))]
    flagged_images = [x for x in visual_predictions if is_phishing_label(x.get("label"))]
    any_prediction = (isinstance(email_prediction, dict) and "label" in email_prediction) or any("label" in x for x in url_predictions) or bool(visual_predictions)
    flagged = email_flagged or bool(flagged_urls) or bool(flagged_images)

    if flagged:
        triage = "HUMAN_REVIEW_REQUIRED"
        recommendation = "At least one available layer flagged a phishing-like indicator. Review the supporting evidence; no automatic blocking was performed."
    elif any_prediction:
        triage = "NO_LAYER_FLAGGED"
        recommendation = "No executed model flagged this input. This is not a guarantee that it is safe; consider layer coverage and contextual indicators before closing the case."
    else:
        triage = "INSUFFICIENT_EVIDENCE"
        recommendation = "No usable model prediction is available. Check layer errors and provide the missing model artifacts or investigation inputs."

    email_label = email_prediction.get("label") if isinstance(email_prediction, dict) else None
    url_count = int(url_result.get("url_count", len(url_predictions)) or 0)
    return {
        "triage": triage,
        "recommendation": recommendation,
        "indicators": {
            "email_label": email_label,
            "email_flagged_phishing": bool(email_flagged),
            "urls_analysed": url_count,
            "phishing_urls_flagged": len(flagged_urls),
            "flagged_url_values": [x.get("url") or x.get("canonical_url") or x.get("canonical") for x in flagged_urls],
            "screenshots_analysed": len(visual_predictions),
            "phishing_screenshots_flagged": len(flagged_images),
            "model_predictions_available": bool(any_prediction),
        },
    }


def request_visual_approval(record: dict[str, Any], initial_assessment: dict[str, Any],
                            args: argparse.Namespace) -> tuple[bool, str]:
    """Require an explicit acceptance before running Visual; default interactive prompt is per case."""
    if args.accept_visual_review:
        return True, "accepted_via_cli_flag"
    if args.skip_visual_review:
        return False, "declined_via_cli_flag"
    if not sys.stdin.isatty():
        return False, "not_accepted_noninteractive"

    title = record.get("subject") or Path(record.get("source") or "Manual input").name
    print("\n" + "-" * 72)
    print(f"Optional Visual analysis available for: {title}")
    print(f"Initial Email/URL assessment: {initial_assessment['triage']}")
    print("Visual will load its model and analyse only the screenshot(s) supplied for this case.")
    try:
        answer = input("Do you accept running the Visual layer for this case? [y/N]: ").strip().lower()
    except EOFError:
        answer = ""
    if answer in {"y", "yes"}:
        return True, "accepted_interactively"
    return False, "declined_or_not_accepted"


def build_record(record: dict[str, Any], runners: LayerRunners, args: argparse.Namespace, record_count: int,
                 evidence_dir: Path) -> dict[str, Any]:
    extracted_urls = extract_urls(record.get("body", ""))
    manual_urls = list(args.urls or [])
    urls = unique_urls(extracted_urls + manual_urls)

    if record.get("input_error"):
        email_result = {"status": "error", "error": record["input_error"]}
    elif record.get("email_present", True):
        email_result = runners.predict_email(record.get("subject", ""), record.get("body", ""))
    else:
        email_result = {"status": "skipped", "reason": "SKIPPED_NO_EMAIL", "message": "No email input was supplied; Email layer remains independent."}

    # URL runs independently of Email: extracted URLs and/or manually supplied URLs are enough.
    url_result = runners.predict_urls(urls)
    images = pair_screenshots(record, args, record_count)

    # Baseline assessment is Email + URL only. Visual is deliberately not loaded/run yet.
    initial_assessment = assess_layers(email_result, url_result)
    if not images:
        visual_result = {"status": "skipped", "reason": "SKIPPED_NO_IMAGE", "message": "No screenshot was supplied or matched."}
        visual_decision = "no_image_available"
        visual_accepted = False
    else:
        visual_accepted, visual_decision = request_visual_approval(record, initial_assessment, args)
        if visual_accepted:
            visual_result = runners.predict_images(images, args.gradcam, evidence_dir)
        else:
            reason = "SKIPPED_REVIEW_NOT_ACCEPTED"
            if visual_decision == "not_accepted_noninteractive":
                reason = "SKIPPED_REVIEW_NOT_ACCEPTED_NONINTERACTIVE"
            visual_result = {
                "status": "skipped",
                "reason": reason,
                "message": "Visual analysis was not run because explicit human acceptance was not provided.",
                "image_count": len(images),
            }

    final_assessment = assess_layers(email_result, url_result, visual_result)
    statuses = [email_result.get("status"), url_result.get("status"), visual_result.get("status")]
    errors = [status for status in statuses if status in {"error", "unavailable", "partial"}]
    return {
        "source": record.get("source"),
        "subject": record.get("subject", ""),
        "email_input_present": bool(record.get("email_present", True)),
        "extracted_urls": extracted_urls,
        "manual_urls": manual_urls,
        "urls_analysed": urls,
        "screenshots": [str(p) for p in images],
        "visual_review": {"accepted": visual_accepted, "decision": visual_decision},
        "initial_assessment": initial_assessment,
        "final_assessment": final_assessment,
        # Keep simple top-level fields for convenient report filtering/backwards compatibility.
        "triage": final_assessment["triage"],
        "recommendation": final_assessment["recommendation"],
        "indicators": final_assessment["indicators"],
        "coverage": {
            "email": email_result.get("status"),
            "url": url_result.get("status"),
            "visual": visual_result.get("status"),
            "partial_or_failed_layer_count": len(errors),
        },
        "layers": {"email": email_result, "url": url_result, "visual": visual_result},
    }


def write_html(report: dict[str, Any], path: Path) -> None:
    def esc(value: Any) -> str:
        return html.escape(str(value if value is not None else ""))
    cards = []
    for index, item in enumerate(report["items"], start=1):
        triage = item.get("triage", "UNKNOWN")
        css = "flagged" if triage == "HUMAN_REVIEW_REQUIRED" else "insufficient" if triage == "INSUFFICIENT_EVIDENCE" else "clear"
        detail_payload = {
            "initial_assessment_email_url_only": item.get("initial_assessment", {}),
            "final_assessment_after_optional_visual": item.get("final_assessment", {}),
            "visual_review": item.get("visual_review", {}),
            "coverage": item.get("coverage", {}),
            "layers": item.get("layers", {}),
        }
        layer_json = json.dumps(detail_payload, ensure_ascii=False, indent=2)
        url_values = item.get("urls_analysed", item.get("extracted_urls", []))
        url_list = "".join(f"<li><code>{esc(url)}</code></li>" for url in url_values) or "<li>None extracted or supplied</li>"
        cards.append(f"""
        <section class="case">
          <div class="case-head"><div><span class="case-num">CASE {index:03d}</span><h2>{esc(item.get('subject') or Path(item.get('source') or 'Direct input').name)}</h2><p class="muted">{esc(item.get('source'))}</p></div><span class="badge {css}">{esc(triage)}</span></div>
          <p>{esc(item.get('recommendation'))}</p>
          <details><summary>URLs analysed ({len(item.get('urls_analysed', item.get('extracted_urls', [])))})</summary><ul>{url_list}</ul></details>
          <details><summary>Assessment, layer outputs &amp; evidence</summary><pre>{esc(layer_json)}</pre></details>
        </section>""")
    total = len(report["items"])
    flagged = sum(x.get("triage") == "HUMAN_REVIEW_REQUIRED" for x in report["items"])
    insufficient = sum(x.get("triage") == "INSUFFICIENT_EVIDENCE" for x in report["items"])
    doc = f"""<!doctype html><html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>Phishing Triage Report</title>
    <style>
    :root{{color-scheme:light;--ink:#182230;--muted:#667085;--line:#e3e8ef;--paper:#f4f6f9;--card:#fff;--navy:#172a46;}}
    *{{box-sizing:border-box}}body{{margin:0;background:var(--paper);color:var(--ink);font:15px/1.55 Segoe UI,Arial,sans-serif}}
    header{{background:var(--navy);color:white;padding:30px max(22px,calc((100vw - 1040px)/2)) 25px}}header h1{{margin:0 0 5px;font-size:27px;letter-spacing:-.3px}}header p{{margin:0;color:#d5ddec}}
    main{{max-width:1040px;margin:24px auto;padding:0 18px}}.metrics{{display:grid;grid-template-columns:repeat(3,minmax(0,1fr));gap:12px;margin-bottom:20px}}.metric,.case{{background:var(--card);border:1px solid var(--line);border-radius:12px}}
    .metric{{padding:16px 18px}}.metric strong{{display:block;font-size:27px}}.metric span{{color:var(--muted);font-size:13px}}
    .case{{padding:20px 22px;margin:14px 0;box-shadow:0 2px 9px rgba(15,23,42,.03)}}.case-head{{display:flex;justify-content:space-between;gap:16px;align-items:flex-start}}.case h2{{font-size:19px;margin:3px 0 2px;overflow-wrap:anywhere}}.case-num{{color:var(--muted);font-size:11px;font-weight:700;letter-spacing:1.3px}}.muted{{color:var(--muted);font-size:12px;overflow-wrap:anywhere}}
    .badge{{border-radius:6px;padding:6px 9px;font-size:11px;font-weight:700;white-space:normal;text-align:center}}.flagged{{background:#fff0ed;color:#9a3412}}.clear{{background:#edf2f7;color:#344054}}.insufficient{{background:#fff5d6;color:#854d0e}}
    details{{border-top:1px solid var(--line);padding:11px 0}}summary{{cursor:pointer;font-weight:600}}pre{{background:#f7f8fa;padding:14px;border-radius:8px;overflow:auto;font-size:12px;white-space:pre-wrap;overflow-wrap:anywhere}}code{{overflow-wrap:anywhere}}footer{{color:var(--muted);font-size:12px;text-align:center;padding:24px}}
    @media(max-width:650px){{.metrics{{grid-template-columns:1fr}}.case-head{{display:block}}.badge{{display:inline-block;margin-top:8px}}}}
    </style></head><body><header><h1>Phishing Detection · Triage Report</h1><p>Human-in-the-loop decision support · Generated {esc(report.get('generated_at'))}</p></header><main>
    <div class="metrics"><div class="metric"><strong>{total}</strong><span>Items analysed</span></div><div class="metric"><strong>{flagged}</strong><span>Require human review</span></div><div class="metric"><strong>{insufficient}</strong><span>Insufficient evidence</span></div></div>
    {''.join(cards) if cards else '<section class="case">No items processed.</section>'}
    </main><footer>Model output is an indicator, not a verdict. No URL was fetched and no blocking action was taken.</footer></body></html>"""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(doc, encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(description="Human-in-the-loop Meta layer for phishing triage.")
    input_group = parser.add_mutually_exclusive_group()
    input_group.add_argument("--email-dir", help="Recursively scan .eml and .txt files in this directory")
    input_group.add_argument("--email-file", help="Scan one .eml or .txt file")
    parser.add_argument("--subject", help="Subject for a single direct email input")
    parser.add_argument("--body", default="", help="Body for a single direct email input")
    parser.add_argument("--urls", nargs="+", help="Manually supplied URL(s); can run URL layer without an email")
    parser.add_argument("--images", nargs="+", help="Screenshot path(s); Visual remains optional and requires approval")
    parser.add_argument("--screenshots-dir", help="Match screenshots by email filename: name.png or name__1.png")
    parser.add_argument("--email-script", help="Explicit path to Email inference script")
    parser.add_argument("--email-model", help="Explicit path to saved Email model")
    parser.add_argument("--email-layer-dir", help="Folder containing Email inference script and feature module")
    parser.add_argument("--url-dir", help="Folder containing URL_Inference / URL_layer_inference.py; defaults to this flat folder, then split folders")
    parser.add_argument("--url-model", help="Path to saved URL model artifact")
    parser.add_argument("--visual-dir", help="Folder containing Visual_Inference.py / visual_layer_inference.py; defaults to this flat folder, then split folders")
    parser.add_argument("--visual-model", help="Path to trained visual checkpoint")
    review_group = parser.add_mutually_exclusive_group()
    review_group.add_argument("--accept-visual-review", action="store_true",
                              help="Explicitly approve Visual analysis for supplied/matched screenshot(s); no prompt")
    review_group.add_argument("--skip-visual-review", action="store_true",
                              help="Do not run Visual and do not prompt, even if screenshots are available")
    parser.add_argument("--gradcam", action="store_true", help="Save Grad-CAM images for approved Visual analyses")
    parser.add_argument("--output-dir", default=str(BASE_DIR / "reports"), help="Directory for JSON and HTML reports")
    args = parser.parse_args()

    has_direct_email = args.subject is not None or bool(args.body)
    has_email_input = bool(args.email_dir or args.email_file or has_direct_email)
    has_manual_investigation_input = bool(args.urls or args.images)
    if not has_email_input and not has_manual_investigation_input:
        parser.error("Provide an email input, --urls, or --images. URL-only and Visual-only runs are supported.")
    if (args.email_dir or args.email_file) and has_direct_email:
        parser.error("Use either an email file/directory or --subject/--body, not both.")
    if args.email_dir and args.urls:
        parser.error("Do not combine --email-dir with --urls: manual URLs would be applied to every email. Use one email input or run URLs separately.")
    if args.email_dir and args.images:
        parser.error("Use --screenshots-dir with --email-dir, or use --images with one email input.")

    try:
        records = collect_inputs(argparse.Namespace(email_dir=args.email_dir, email_file=args.email_file,
                                                    subject=args.subject, body=args.body))
    except Exception as exc:
        parser.error(f"Could not load input: {type(exc).__name__}: {exc}")
    if not records:
        mode = "manual_url_and_visual_input" if args.urls and args.images else "manual_url_input" if args.urls else "manual_visual_input"
        records = [{"source": mode, "name": "manual_input", "subject": "", "body": "", "email_present": False}]

    runners = LayerRunners(args)
    output_dir = Path(args.output_dir).expanduser().resolve()
    evidence_dir = output_dir / "evidence"
    items = [build_record(record, runners, args, len(records), evidence_dir) for record in records]
    report = {
        "project": "Phishing Detection Capstone",
        "generated_at": datetime.now().astimezone().isoformat(timespec="seconds"),
        "mode": "independent Email and URL layers; Visual requires explicit human acceptance; no automatic blocking",
        "items_processed": len(items),
        "summary": {
            "human_review_required": sum(x["triage"] == "HUMAN_REVIEW_REQUIRED" for x in items),
            "no_layer_flagged": sum(x["triage"] == "NO_LAYER_FLAGGED" for x in items),
            "insufficient_evidence": sum(x["triage"] == "INSUFFICIENT_EVIDENCE" for x in items),
            "visual_analyses_run": sum(x["coverage"]["visual"] in {"ok", "partial", "error"} for x in items),
        },
        "items": items,
        "notes": [
            "Email and URL inference are independent: Email runs with email content; URL runs when extracted or manually supplied URLs are available.",
            "No URL is a skipped condition (SKIPPED_NO_URL), not a pipeline failure; URL can also be run with --urls without an email.",
            "Visual checkpoint loading and inference occur only after explicit approval, via interactive confirmation or --accept-visual-review.",
            "Visual can be run independently using --images with no email input, after human approval.",
            "Layer probabilities are not combined into a single numeric risk score because models are not jointly calibrated; triage and indicators are recalculated from available layer outputs.",
            "NO_LAYER_FLAGGED is not a guarantee that an email or website is safe.",
            "This prototype does not fetch or open URLs and does not automatically block anything.",
            "The email body is used for inference but is not written into the report by default.",
        ],
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    json_path = output_dir / "meta_report.json"
    html_path = output_dir / "meta_report.html"
    json_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    write_html(report, html_path)

    print("\n" + "=" * 72)
    print("PHISHING DETECTION — META LAYER REPORT")
    print("=" * 72)
    print(f"Items processed : {len(items)}")
    print(f"Review required : {report['summary']['human_review_required']}")
    print(f"No layer flagged: {report['summary']['no_layer_flagged']}")
    print(f"Insufficient    : {report['summary']['insufficient_evidence']}")
    print(f"Visual analyses : {report['summary']['visual_analyses_run']}")
    for i, item in enumerate(items, start=1):
        print("-" * 72)
        print(f"[{i}/{len(items)}] {Path(item.get('source') or '').name or 'Manual input'}")
        if item.get("subject"):
            print(f"Subject           : {item['subject']}")
        print(f"Initial assessment: {item['initial_assessment']['triage']} (Email + URL)")
        print(f"Final assessment  : {item['triage']}")
        email_layer = item["layers"]["email"]
        if email_layer.get("status") == "ok":
            result = email_layer["result"]
            print(f"Email             : {str(result.get('label', 'unknown')).upper()} | p={float(result.get('probability', 0)):.4f} | threshold={float(result.get('threshold', 0)):.4f}")
        else:
            print(f"Email             : {email_layer.get('status')} ({email_layer.get('reason', email_layer.get('error', ''))})")
        url_layer = item["layers"]["url"]
        if url_layer.get("status") == "ok":
            results = url_layer.get("results", [])
            flagged_urls = sum(is_phishing_label(x.get("label")) for x in results if isinstance(x, dict))
            print(f"URL               : {flagged_urls}/{len(results)} flagged phishing")
        else:
            print(f"URL               : {url_layer.get('status')} ({url_layer.get('reason', url_layer.get('error', ''))})")
        visual = item["layers"]["visual"]
        if visual.get("status") in {"ok", "partial"}:
            preds = [x for x in visual.get("results", []) if "label" in x]
            flagged_images = sum(is_phishing_label(x.get("label")) for x in preds)
            print(f"Visual            : {flagged_images}/{len(preds)} screenshots flagged phishing")
        else:
            print(f"Visual            : {visual.get('status')} ({visual.get('reason', visual.get('error', ''))})")
        print(f"Visual accepted   : {item['visual_review']['accepted']} ({item['visual_review']['decision']})")
        print(f"Indicators        : {json.dumps(item['indicators'], ensure_ascii=False)}")
        print(f"Note              : {item['recommendation']}")
    print("-" * 72)
    print(f"JSON report : {json_path}")
    print(f"HTML report : {html_path}")
    print("No URL was fetched. No blocking action was taken.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
