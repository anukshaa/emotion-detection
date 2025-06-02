import os
# Define the paths to the train and test directories
train_dir = r'C:\Users\Anshika\OneDrive\Desktop\mood_music_recommendation\fer2013\train'
test_dir = r'C:\Users\Anshika\OneDrive\Desktop\mood_music_recommendation\fer2013\test'

# Function to list files in a directory
def list_files(directory):
    for root, dirs, files in os.walk(directory):
        for name in files:
            print(os.path.join(root, name))

print("Train Directory Contents:")
list_files(train_dir)

print("\nTest Directory Contents:")
list_files(test_dir)
