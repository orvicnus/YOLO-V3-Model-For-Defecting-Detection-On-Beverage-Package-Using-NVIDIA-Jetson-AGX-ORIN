#Get the value of each one label
label_files = [f for f in os.listdir(label_train_dir) if f.endswith('.txt')]

# Take 10 data randomly
label_files = label_files[:5]

for label_file in label_files:
    label_path = os.path.join(label_train_dir, label_file)
    with open(label_path, 'r') as f:
        lines = f.readlines()
        print(f"Label untuk file {label_file}:")
        for line in lines:
            print(line.strip())
        print()  # Untuk pemisah antar file label

#Checking The Data Path
print(image_train_dir)
print(label_train_dir)
print(image_test_dir)
print(label_test_dir)