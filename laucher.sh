#!/bin/bash
echo "Welcome to the program launcher"
echo "Be sure to have the requirements installed"
PS3='Choose what you would like to do today: '
option=("TFTEST" "More" "Quit")
select fav in "${option[@]}"; do
    case $fav in
        "TFTEST")
            echo "Testing out for compatible GPUs (AMD ROC GPUs won't appear unless Tensorflow compiled to use ROC"
            python Python/TF_test_GPU.py
            ;;
        "More")
            echo "Come back later for more options"
            echo "thank you, see you again soon"
	    exit
            ;;
        "Quit")
            echo "User requested exit"
            exit
            ;;
        *) echo "invalid option $REPLY";;
    esac
done
