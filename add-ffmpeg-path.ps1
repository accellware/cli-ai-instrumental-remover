[Environment]::SetEnvironmentVariable(
    "PATH",
    [Environment]::GetEnvironmentVariable("PATH", "User") + ";C:\tools\ffmpeg-8.1.1-full_build-shared\bin",
    "User"
)
Write-Host "Added FFmpeg to user PATH. Restart your terminal to apply."
