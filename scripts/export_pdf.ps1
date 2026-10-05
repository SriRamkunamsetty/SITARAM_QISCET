# Export docs/presentation.pptx to docs/presentation.pdf using Microsoft PowerPoint (Windows).
#   powershell -File scripts/export_pdf.ps1
param(
    [string]$Source = "$PSScriptRoot\..\docs\presentation.pptx",
    [string]$Target = "$PSScriptRoot\..\docs\presentation.pdf"
)

$srcPath = (Resolve-Path $Source).Path
$dstPath = [System.IO.Path]::GetFullPath($Target)

$app = New-Object -ComObject PowerPoint.Application
try {
    $pres = $app.Presentations.Open($srcPath, $true, $false, $false)  # read-only, no window
    $pres.SaveAs($dstPath, 32)                                         # 32 = ppSaveAsPDF
    $pres.Close()
    Write-Host "wrote $dstPath"
}
finally {
    $app.Quit()
}
