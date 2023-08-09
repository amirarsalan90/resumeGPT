#!/bin/bash

# set the name of the virtual environment
VENV_NAME="resumegptenv"

# check if the virtual environment already exists
if [ ! -d "$VENV_NAME" ]
then
    echo "Creating virtual environment..."
    python3 -m venv $VENV_NAME

    echo "Activating virtual environment..."
    source $VENV_NAME/bin/activate

    echo "Installing Python dependencies..."
    pip install -r requirements.txt

else
    echo "Activating existing virtual environment..."
    source $VENV_NAME/bin/activate
fi

echo "Running Python script..."
streamlit run main.py