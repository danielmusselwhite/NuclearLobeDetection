# Prompt the user to enter the path to the directory containing the .txt files
$directoryPath = Read-Host -Prompt "Enter the path to the directory containing the .txt files"

# Check if the directory exists
if (-Not (Test-Path -Path $directoryPath)) {
    Write-Output "The directory path '$directoryPath' does not exist. Please check the path and try again."
    exit
}

# Initialize a hashtable for the overall count across all files
$overallCount = @{}

# Get all .txt files in the directory
$files = Get-ChildItem -Path $directoryPath -Filter *.txt

# Check if there are any .txt files in the directory
if ($files.Count -eq 0) {
    Write-Output "No .txt files found in the directory '$directoryPath'."
    exit
}

# Loop through each file
foreach ($file in $files) {
    # Initialize a hashtable for the individual file count
    $fileCount = @{}

    # Read the content of the file
    $lines = Get-Content -Path $file.FullName

    # Process each line in the file
    foreach ($line in $lines) {
        # Extract the first column (split by space and take the first element)
        $number = ($line -split '\s+' | Select-Object -First 1) -as [int]

        # Ensure $number is a valid integer
        if ($number -ne $null) {
            # Update the count for the current file
            if ($fileCount.ContainsKey($number)) {
                $fileCount[$number]++
            } else {
                $fileCount[$number] = 1
            }

            # Update the overall count
            if ($overallCount.ContainsKey($number)) {
                $overallCount[$number]++
            } else {
                $overallCount[$number] = 1
            }
        }
    }

    # Output the count for the current file with sorted keys
    Write-Output "Counts for file $($file.Name):"
    foreach ($key in ($fileCount.Keys | Sort-Object)) {
        Write-Output "$($key+1) : $($fileCount[$key])"
    }
    Write-Output ""
}

# Output the overall count across all files with sorted keys
Write-Output "Overall Counts across all files:"
foreach ($key in ($overallCount.Keys | Sort-Object)) {
    Write-Output "$($key+1) : $($overallCount[$key])"
}
