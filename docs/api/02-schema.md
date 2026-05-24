# Schema (UI Widget Definitions)

The schema endpoints return the complete widget definitions for every parameter panel. The React UI reads these once on startup and renders the correct widget type for each parameter — no hardcoding required.

---

## GET /api/schema/control

Returns the global settings panel schema (provider, threads, detector, webcam, virtual cam, WebRTC ports, etc.).

## GET /api/schema/parameters/swap

Returns the face-swap parameter panel schema (swapper model, DFM settings, keypoint adjustments, masking, color correction, etc.).

## GET /api/schema/parameters/common

Returns the common parameter panel schema (face restorer, expression restorer).

## GET /api/schema/parameters/face-editor

Returns the LivePortrait face-editor panel schema (head pose, expression sliders, makeup).

---

All four endpoints return the same shape:

**Response**

```json
{
  "widgets": [
    {
      "widget_name": "SwapModelSelection",
      "section": "Swapper",
      "level": 1,
      "label": "Swapper Model",
      "widget_type": "selection",
      "default": "Inswapper128",
      "options": [
        "Inswapper128",
        "InStyleSwapper256 Version A",
        "InStyleSwapper256 Version B",
        "InStyleSwapper256 Version C",
        "DeepFaceLive (DFM)",
        "SimSwap512",
        "GhostFace-v1",
        "GhostFace-v2",
        "GhostFace-v3",
        "CSCS"
      ],
      "min_value": null,
      "max_value": null,
      "step": null,
      "decimals": null,
      "help": "Choose which swapper model to use for face swapping.",
      "parent_toggle": null,
      "required_toggle_value": null,
      "parent_selection": null,
      "required_selection_value": null,
      "width": null
    },
    {
      "widget_name": "FaceRestorerEnableToggle",
      "section": "Face Restorer",
      "level": 1,
      "label": "Enable Face Restorer",
      "widget_type": "toggle",
      "default": false,
      "options": null,
      ...
    },
    {
      "widget_name": "FaceRestorerBlendSlider",
      "section": "Face Restorer",
      "level": 2,
      "label": "Blend",
      "widget_type": "slider",
      "default": 100,
      "min_value": 0,
      "max_value": 100,
      "step": 1,
      "parent_toggle": "FaceRestorerEnableToggle",
      "required_toggle_value": true,
      ...
    }
  ]
}
```

### Widget descriptor fields

| Field | Type | Description |
|---|---|---|
| `widget_name` | string | The key used in `parameters` or `control` dicts. |
| `section` | string | Group heading (e.g. `"Swapper"`, `"Face Restorer"`). |
| `level` | int | Indent depth: `1` = top-level, `2` = child of a toggle/selection, `3` = grandchild. |
| `label` | string | Human-readable display label. |
| `widget_type` | enum | `"toggle"` · `"slider"` · `"decimal_slider"` · `"selection"` · `"text"` |
| `default` | any | Default value. |
| `options` | string[] \| null | Dropdown options for `selection` widgets. |
| `min_value` | number \| null | Minimum for slider widgets. |
| `max_value` | number \| null | Maximum for slider widgets. |
| `step` | number \| null | Step size for slider widgets. |
| `decimals` | int \| null | Decimal places for `decimal_slider` widgets. |
| `help` | string | Tooltip / description text. |
| `parent_toggle` | string \| null | Name of a toggle widget this widget depends on. Show only when that toggle matches `required_toggle_value`. |
| `required_toggle_value` | bool \| null | Required state of `parent_toggle` for this widget to be visible. |
| `parent_selection` | string \| null | Name of a selection widget this widget depends on. |
| `required_selection_value` | string \| null | Required value of `parent_selection` for this widget to be visible. |
| `width` | int \| null | Suggested fixed width in pixels (for text inputs). |

### Rendering logic

```tsx
function renderWidget(w: WidgetDescriptor, value: any, onChange: (v: any) => void) {
  // Visibility: hide if parent toggle/selection doesn't match
  if (w.parent_toggle && controlValues[w.parent_toggle] !== w.required_toggle_value) return null;
  if (w.parent_selection && controlValues[w.parent_selection] !== w.required_selection_value) return null;

  switch (w.widget_type) {
    case 'toggle':         return <Switch checked={value} onCheckedChange={onChange} />;
    case 'slider':         return <Slider min={w.min_value} max={w.max_value} step={w.step} value={[value]} onValueChange={([v]) => onChange(v)} />;
    case 'decimal_slider': return <Slider min={w.min_value} max={w.max_value} step={w.step} value={[value]} onValueChange={([v]) => onChange(parseFloat(v.toFixed(w.decimals)))} />;
    case 'selection':      return <Select value={value} onValueChange={onChange} options={w.options} />;
    case 'text':           return <Input value={value} onChange={e => onChange(e.target.value)} />;
  }
}
```

---

## GET /api/schema/dfm-models

Returns available DeepFaceLab DFM model files scanned from `model_assets/dfm_models/`.

**Response**

```json
{
  "my_model.dfm": "/absolute/path/to/model_assets/dfm_models/my_model.dfm",
  "another.onnx": "/absolute/path/to/model_assets/dfm_models/another.onnx"
}
```

Returns `{}` if no DFM models are installed. The keys are used as values for the `DFMModelSelection` parameter.
