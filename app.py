import os
import subprocess
import shutil
import tempfile
from PIL import Image
import streamlit as st
from deepface import DeepFace
import matplotlib.pyplot as plt


# Function to detect age using DeepFace
def detect_age(image_path):
    """
    Detects the age of the person in the image using DeepFace.

    :param image_path: Path to the input image
    :return: Detected age (integer) or None if detection fails
    """
    try:
        analysis = DeepFace.analyze(img_path=image_path, actions=['age'], enforce_detection=True)
        if isinstance(analysis, list):
            # If multiple faces are detected, take the first one
            age = analysis[0]['age']
        else:
            # Single face detected
            age = analysis['age']
        return age
    except Exception as e:
        st.error(f"Error in age detection: {e}")
        return None


# Function to run StarGAN's test mode
def run_stargan(stargan_dir, transform_choice, image_filename):
    """
    Runs StarGAN's test mode to perform the age transformation.

    :param stargan_dir: Path to the StarGAN directory
    :param transform_choice: 'older' or 'younger'
    :param image_filename: Filename of the input image
    :return: Path to the transformed image or None if failed
    """
    # Define paths
    images_dir = os.path.join(stargan_dir, 'stargan_celeba_256', 'images')
    results_dir = os.path.join(stargan_dir, 'stargan_celeba_256', 'results')
    labels_dir = os.path.join(stargan_dir, 'stargan_celeba_256', 'labels')
    attr_file_dir = os.path.join(stargan_dir, 'stargan_celeba_256', 'data', 'celeba')
    attr_file_path = os.path.join(attr_file_dir, 'list_attr_celeba.txt')

    # Ensure directories exist
    os.makedirs(images_dir, exist_ok=True)
    os.makedirs(results_dir, exist_ok=True)
    os.makedirs(labels_dir, exist_ok=True)
    os.makedirs(attr_file_dir, exist_ok=True)

    # Copy and resize the input image to 'images_dir/input.jpg'
    input_image_path = os.path.join(images_dir, 'input.jpg')
    try:
        img = Image.open(os.path.join(stargan_dir, image_filename)).convert('RGB')
    except FileNotFoundError:
        st.error(f"Input image {image_filename} not found in {stargan_dir}.")
        return None
    img = img.resize((256, 256), Image.BICUBIC)
    img.save(input_image_path)
    st.info(f"Input image saved to {input_image_path}")

    # Create a minimal 'list_attr_celeba.txt' file
    # StarGAN expects the first line to be the number of images
    # The second line to be the attribute names
    # Subsequent lines: image_name followed by attribute values

    # For simplicity, we'll set other attributes to '0' and 'Young' based on transform_choice
    other_attrs = ['0', '0', '0', '0']  # Black_Hair, Blond_Hair, Brown_Hair, Male
    if transform_choice == 'older':
        young_attr = '0'  # Not Young
    else:
        young_attr = '1'  # Young
    target_attrs = other_attrs + [young_attr]
    label_content = '1\nBlack_Hair Blond_Hair Brown_Hair Male Young\ninput.jpg ' + ' '.join(target_attrs) + '\n'

    # Write the attribute file
    with open(attr_file_path, 'w') as f:
        f.write(label_content)
    st.info(f"Attribute file created at {attr_file_path} with attributes: {' '.join(target_attrs)}")

    # Path to pre-trained Generator checkpoint
    G_checkpoint = '200000-G.ckpt'

    # Run StarGAN's test_stargan.py
    command = [
        'python', 'test_stargan.py',
        '--transform_choice', transform_choice,
        '--model_save_dir', os.path.join('stargan_celeba_256', 'models'),
        '--G_checkpoint', G_checkpoint,
        '--input_image', image_filename,
        '--result_dir', os.path.join('stargan_celeba_256', 'results'),
        '--c_dim', '5',
        '--g_conv_dim', '64',
        '--g_repeat_num', '6',
        '--image_size', '256'
    ]

    # Convert backslashes to forward slashes for compatibility (especially on Windows)
    command = [str(arg).replace('\\', '/') if isinstance(arg, str) else arg for arg in command]

    st.info(f"Running StarGAN transformation using checkpoint '{G_checkpoint}'...")
    try:
        # Execute the command
        result = subprocess.run(command, cwd=stargan_dir, capture_output=True, text=True)
        if result.returncode != 0:
            st.error(f"Error running StarGAN: {result.stderr}")
            return None
    except Exception as e:
        st.error(f"Exception while running StarGAN: {e}")
        return None

    # Locate the transformed image
    transformed_image_path = os.path.join(results_dir, 'transformed_image.jpg')
    if not os.path.exists(transformed_image_path):
        st.error("Transformed image not found in the results directory.")
        return None

    st.success("Image transformation completed successfully.")
    return transformed_image_path


# Streamlit App Layout
def main():
    st.set_page_config(page_title="Age Detection and Transformation App", layout="wide")
    st.title("🧓👶 Age Detection and Transformation App")
    st.write("Upload an image, detect the age of the person, and transform the image to appear older or younger using StarGAN.")

    # Sidebar for instructions
    with st.sidebar:
        st.header("📚 Instructions")
        st.markdown("""
            1. **Upload an Image:** Click on the "Browse files" button and select an image in JPG, JPEG, or PNG format.
            2. **Detect Age:** The app will automatically detect the age of the person in the image using DeepFace.
            3. **Choose Transformation:** Select whether you want to make the person appear **older** or **younger**.
            4. **Transform Image:** Click the "Transform Image" button to perform the transformation.
            5. **View Results:** The transformed image will be displayed below.
        """)

    # File uploader
    uploaded_file = st.file_uploader("📤 **Choose an image...**", type=["jpg", "jpeg", "png"])

    if uploaded_file is not None:
        # Save the uploaded image to a temporary file
        with tempfile.NamedTemporaryFile(delete=False, suffix='.jpg') as tmp_file:
            tmp_file.write(uploaded_file.read())
            tmp_image_path = tmp_file.name

        # Display the uploaded image in the main area
        st.subheader("### 📷 Uploaded Image")
        st.image(tmp_image_path, caption='Uploaded Image', use_column_width=True)

        # Detect age
        st.subheader("### 🔍 Age Detection")
        with st.spinner("Detecting age..."):
            age = detect_age(tmp_image_path)
        if age is not None:
            st.success(f"**Detected Age:** {age} years old")
        else:
            st.error("Failed to detect age.")
            return

        # Transformation choice
        st.subheader("### 🔄 Choose Transformation")
        transform_choice = st.radio("Select the desired transformation:", ('👵 Make Older', '👶 Make Younger'))

        # Map the choices to actual transformation commands
        transform_map = {
            '👵 Make Older': 'older',
            '👶 Make Younger': 'younger'
        }

        # Transform Image button
        if st.button("🔄 Transform Image"):
            with st.spinner("Transforming image..."):
                # Define paths
                stargan_dir = os.path.abspath(os.getcwd())  # Assuming the app is run from the 'stargan_project' directory

                # Save the uploaded image to the stargan directory with a known filename
                image_filename = 'uploaded_input.jpg'
                destination_image_path = os.path.join(stargan_dir, image_filename)
                shutil.copyfile(tmp_image_path, destination_image_path)

                # Run StarGAN
                transformed_image_path = run_stargan(stargan_dir, transform_map[transform_choice], image_filename)

                if transformed_image_path:
                    # Display the transformed image
                    st.subheader("### 🎨 Transformed Image")
                    st.image(transformed_image_path, caption='Transformed Image', use_column_width=True)
                else:
                    st.error("Image transformation failed.")

            # Optional: Clean up temporary files after processing
            # Uncomment the lines below if you wish to enable clean-up
            # os.remove(tmp_image_path)
            # os.remove(destination_image_path)
    else:
        st.info("📌 Please upload an image to get started.")


if __name__ == "__main__":
    main()
