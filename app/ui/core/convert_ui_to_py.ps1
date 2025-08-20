# Define relative paths
$UI_FILE = Join-Path $PSScriptRoot "MainWindow.ui"
$PY_FILE = Join-Path $PSScriptRoot "main_window.py"
$QRC_FILE = Join-Path $PSScriptRoot "media.qrc"
$RCC_PY_FILE = Join-Path $PSScriptRoot "media_rc.py"

# Run PySide6 commands
pyside6-uic $UI_FILE -o $PY_FILE
pyside6-rcc $QRC_FILE -o $RCC_PY_FILE

# Define search and replace strings
$searchString = "import media_rc"
$replaceString = "from app.ui.core import media_rc"

# Read the file content
$content = Get-Content $PY_FILE -Raw

# Perform the replacement
$content = $content -replace $searchString, $replaceString

# Write the modified content back to the file
$content | Set-Content $PY_FILE -NoNewline

Write-Host "Replacement complete."
