# Controls

## Mouse

- Left click: positive point
- Right click: negative point
- Mouse wheel: zoom
- Middle-button drag: pan
- `Shift + Left drag`: pan
- Box mode + left drag: create box prompt
- Brush mode + left drag: draw mask
- Brush mode + right drag: erase mask
- Brush mode + `Ctrl + Left click` twice: draw a straight line between two points

## Keyboard

- `Enter`: confirm current instance
- `S`: save current image annotations
- `Space` / `Right`: next image
- `Left`: previous image
- `B`: toggle box mode
- `E`: toggle edit-mask mode
- `T`: revert current mask to SAM prediction
- `Backspace`: undo last confirmed instance
- `U`: undo last point
- `R`: reset current points/mask
- `[` / `]`: brush radius - / +
- `N` / `P` / `1~9`: switch class
- `Esc`: cancel current box drag or brush line preview

Note: shortcuts are customizable from the top-left Settings (gear) panel.

## UI Actions

- `Flag` button: toggle flag for current image
- `Overview` button: open thumbnail wall and jump to image
- `Brush` button: switch brush type between Add and Erase
- `Settings` (gear): open shortcut binding dialog

## Left Panel Settings

- View Adjustment sliders: brightness / contrast / saturation
- `Reset` button restores view filters to defaults
- View filters are display-only (do not change saved masks/labels)
- View settings are stored in browser local storage

## Brush Edit Mode

- Press `E` to toggle edit mode for the current mask
- In edit mode, left-drag paints mask (add), right-drag erases mask
- `[` / `]` adjusts brush radius
- `T` resets current edited mask to the latest SAM output
- One continuous mouse drag counts as one brush stroke for undo behavior
- You can use brush-only flow for tiny defects when bbox/points are not ideal

## Annotation Stats Panel

- Located under Confirmed Instances
- Shows per-class `Count`, `Avg score`, `Avg area`, and `Area range`
- Helps detect annotation drift during long labeling sessions

## Notes

- `Enter` confirms in memory only
- `S` writes outputs to disk
- Confirmed instances restore from autosave when revisiting image
- Flag restore is disabled by default; use `--restore-flags` to enable
