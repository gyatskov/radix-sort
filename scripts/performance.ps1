write "Starting Testloop..."

for ($i=25; $i -gt 0; $i--) {
    $p = [math]::Pow(2,$i)
    write "..\build\Debug\Assignment.exe --num-elements $p --perf-csv-to-stdout > ..\Performance\input_test_2_$i.txt 2>&1"
    ..\build\Debug\Assignment.exe --num-elements $p --perf-csv-to-stdout > ..\Performance\input_test_2_$i.txt 2>&1
}