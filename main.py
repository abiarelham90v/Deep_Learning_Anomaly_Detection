'''This is the repo which contains the original code to the WACV 2021 paper
"Same Same But DifferNet: Semi-Supervised Defect Detection with Normalizing Flows"
by Marco Rudolph, Bastian Wandt and Bodo Rosenhahn.
For further information contact Marco Rudolph (rudolph@tnt.uni-hannover.de)'''


import config as c
from train import train
from utils import load_datasets, make_dataloaders

def main():
    for class_name in c.class_names:
        print(f"Training model for class: {class_name}")
        
        # Update the model name for the current class
        c.modelname = c.modelname_template.format(class_name)
        
        # Load datasets for the current class
        train_set, test_set = load_datasets(c.dataset_path, class_name)
        train_loader, test_loader = make_dataloaders(train_set, test_set)
        
        # Train the model for the current class
        model = train(train_loader, test_loader)
        
        print(f"Finished training for class: {class_name}\n")

if __name__ == "__main__":
    main()