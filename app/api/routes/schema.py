"""
GET /api/schema/control
GET /api/schema/parameters/swap
GET /api/schema/parameters/common
GET /api/schema/parameters/face-editor
GET /api/schema/dfm-models
"""
from __future__ import annotations

from typing import Any, Dict, List

from fastapi import APIRouter

from app.api.schemas import SchemaResponse, WidgetDescriptor
from app.helpers.miscellaneous import get_dfm_models_data

router = APIRouter(prefix="/api/schema", tags=["schema"])


def _layout_to_descriptors(layout_data: Dict[str, Any]) -> List[WidgetDescriptor]:
    """
    Convert a *_layout_data.py dict into a list of WidgetDescriptor objects.
    Callable 'options' and 'default' values are resolved at request time.
    exec_function references are stripped — the React UI uses event names instead.
    """
    descriptors: List[WidgetDescriptor] = []
    for section, widgets in layout_data.items():
        for widget_name, cfg in widgets.items():
            # Resolve callable options/defaults (e.g. DFM model list)
            options = cfg.get("options")
            if callable(options):
                options = options()
            if options is not None:
                options = [str(o) for o in options]

            default = cfg.get("default")
            if callable(default):
                default = default()

            # Infer widget type from descriptor shape
            if options is not None:
                wtype = "selection"
            elif "decimals" in cfg:
                wtype = "decimal_slider"
            elif "min_value" in cfg and "max_value" in cfg:
                wtype = "slider"
            elif isinstance(default, bool):
                wtype = "toggle"
            else:
                wtype = "text"

            # Parse numeric bounds (stored as strings in layout_data)
            def _num(v: Any) -> Any:
                if v is None:
                    return None
                try:
                    return float(v) if "." in str(v) else int(v)
                except (ValueError, TypeError):
                    return None

            descriptors.append(WidgetDescriptor(
                widget_name=widget_name,
                section=section,
                level=cfg.get("level", 1),
                label=cfg.get("label", widget_name),
                widget_type=wtype,
                default=default,
                options=options,
                min_value=_num(cfg.get("min_value")),
                max_value=_num(cfg.get("max_value")),
                step=_num(cfg.get("step")),
                decimals=cfg.get("decimals"),
                help=cfg.get("help", ""),
                parent_toggle=cfg.get("parentToggle"),
                required_toggle_value=cfg.get("requiredToggleValue"),
                parent_selection=cfg.get("parentSelection"),
                required_selection_value=cfg.get("requiredSelectionValue"),
                width=cfg.get("width"),
            ))
    return descriptors


@router.get("/control", response_model=SchemaResponse)
def schema_control():
    """Return the global settings (control) widget schema."""
    from app.ui.widgets.settings_layout_data import SETTINGS_LAYOUT_DATA
    return SchemaResponse(widgets=_layout_to_descriptors(SETTINGS_LAYOUT_DATA))


@router.get("/parameters/swap", response_model=SchemaResponse)
def schema_swap():
    """Return the face-swap parameter widget schema."""
    from app.ui.widgets.swapper_layout_data import SWAPPER_LAYOUT_DATA
    return SchemaResponse(widgets=_layout_to_descriptors(SWAPPER_LAYOUT_DATA))


@router.get("/parameters/common", response_model=SchemaResponse)
def schema_common():
    """Return the common (restorer / expression) parameter widget schema."""
    from app.ui.widgets.common_layout_data import COMMON_LAYOUT_DATA
    return SchemaResponse(widgets=_layout_to_descriptors(COMMON_LAYOUT_DATA))


@router.get("/parameters/face-editor", response_model=SchemaResponse)
def schema_face_editor():
    """Return the LivePortrait face-editor parameter widget schema."""
    from app.ui.widgets.face_editor_layout_data import FACE_EDITOR_LAYOUT_DATA
    return SchemaResponse(widgets=_layout_to_descriptors(FACE_EDITOR_LAYOUT_DATA))


@router.get("/dfm-models")
def schema_dfm_models() -> Dict[str, str]:
    """Return available DFM model files: {filename: absolute_path}."""
    return get_dfm_models_data()
