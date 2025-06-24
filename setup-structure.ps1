# setup-structure.ps1
$root = "tetrio_ai_trainer"

# Define the directories
$dirs = @(
    "$root/engine",
    "$root/ai",
    "$root/sessions",
    "$root/models"
)

# Create each directory with a .gitkeep file
foreach ($dir in $dirs) {
    New-Item -ItemType Directory -Path $dir -Force | Out-Null
    New-Item -ItemType File -Path "$dir/.gitkeep" -Force | Out-Null
}

# Create key Python files
$files = @(
    "$root/engine/core.py",
    "$root/engine/piece.py",
    "$root/ai/agent.py",
    "$root/ai/trainer.py",
    "$root/sessions/runner.py",
    "$root/sessions/feedback.py",
    "$root/main.py",
    "$root/requirements.txt"
)

foreach ($file in $files) {
    New-Item -ItemType File -Path $file -Force | Out-Null
}

# Write requirements.txt content
@"
numpy
matplotlib
torch        # or tensorflow if preferred
"@ | Set-Content -Path "$root/requirements.txt"

Write-Host "Project structure and requirements.txt created in '$root'"
