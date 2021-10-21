#!/bin/bash
echo "Welcome to the program launcher"
echo "Be sure to have the requirements installed"
PS3='Choose what you would like to do today: '
option=("Pre-Processing" "Training" "Quit")
select fav in "${option[@]}"; do
    case $fav in
        "Pre-Processing")
            PS3='Choose your what pre-processing you would like to do:'
            option=("TFTEST" "Benchmark" "Convert dataset to jpeg" "Convert video to jpeg" "Normalise dataset" "Quit")
            select fav in "${option[@]}"; do
               case $fav in
                    "TFTEST")
                        echo "Testing out for compatible GPUs (AMD ROC GPUs won't appear unless Tensorflow compiled to use ROC"
                        python -c "from code.tester import tf_test ; tf_test()" 
                        ;;
                    "Benchmark")
                       echo "Benchmarking Tensorflow Setup"
                        python -c "from code.tester import benchmark ; benchmark()"
                        ;;
                    "Convert dataset to jpeg")
                        echo "Ensure the dataset path is correct"
                        echo "Beginning conversion"
                        python -c 'from code.convert import convert; convert()'  
                       ;;
                    "Convert video to jpeg")
                        echo "Ensure the video is in the correct place"
                        echo "Beginning conversion"
                        python -c 'from code.convert import convertVideoToImage; convertVideoToImage()'
                    ;;
                    "Normalise dataset")
                        echo "Ensure the dataset has been resized and imported properly"
                        echo "Beginning normalisation"
                        python -c "from code.convert import * ; normalise()"
                    ;;
                    "Quit")
                        echo "User requested exit"
                        exit 
                        ;;
                esac
        "Training")
            echo "Section under construction"
            
            ;;
        "Quit")
            echo "User requested exit"
            exit
            ;;
        *) echo "invalid option $REPLY";;
    esac
done
