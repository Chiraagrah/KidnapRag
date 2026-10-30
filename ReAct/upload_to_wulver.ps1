# Uploads the MIS-diagnostic files to Wulver.
# Run from PowerShell:
#   powershell -ExecutionPolicy Bypass -File upload_to_wulver.ps1
# You'll get a UCID password + DUO prompt per directory (4 total).

$ErrorActionPreference = "Stop"

$Base   = "C:\Users\chira\Desktop\NJIT Courses\Sem3\MastersProject\KidnapRag\KidnapRAG\ReAct"
$Remote = "cs757@wulver.njit.edu:/project/ss797/cs757/KidnapRAG/ReAct"

Write-Host "1/4  src/reliability_mis.py ..."
scp "$Base\src\reliability_mis.py" "$Remote/src/"

Write-Host "2/4  ReAct/attack_react_mis.py + e5_env.py ..."
scp "$Base\ReAct\attack_react_mis.py" "$Base\ReAct\e5_env.py" "$Remote/ReAct/"

Write-Host "3/4  scripts/plot_mis.py ..."
scp "$Base\scripts\plot_mis.py" "$Remote/scripts/"

Write-Host "4/4  run_mis_clean.sh ..."
scp "$Base\run_mis_clean.sh" "${Remote}/"

Write-Host "`nDone. Files are under /project/ss797/cs757/KidnapRAG/ReAct/ on Wulver." -ForegroundColor Green
