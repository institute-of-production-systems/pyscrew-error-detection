@echo off
echo Starting PyScrew Error Detection Experiments
echo ============================================

REM Activate virtual environment (adjust path as needed)
call venv\Scripts\activate.bat

REM Run multiple experiments
echo.
echo [1/6] Running s04 binary vs ref with paper models...
python main.py --scenario s04 --sampling binary_vs_ref --modeling paper

echo.
echo [2/6] Running s04 multiclass with groups...
python main.py --scenario s04 --sampling multiclass_with_groups --modeling paper

echo.
echo [3/6] Running s05 binary for extremes...
python main.py --scenario s05 --sampling binary_for_extremes --modeling paper

echo.
echo [4/6] Running s05 multiclass within groups...
python main.py --scenario s05 --sampling multiclass_within_groups --modeling paper

echo.
echo [5/6] Running s06 binary for extremes...
python main.py --scenario s06 --sampling binary_for_extremes --modeling paper

echo.
echo [6/6] Running s06 multiclass within groups...
python main.py --scenario s06 --sampling multiclass_within_groups --modeling paper

echo.
echo All experiments completed!
echo View results at: http://localhost:5000
pause