from flask import Flask, render_template, request, abort
from werkzeug.utils import secure_filename
import cv2
import os
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from matplotlib.gridspec import GridSpecFromSubplotSpec
from sklearn.cluster import KMeans
import imutils
import matplotlib.colors as mcolors

### PENTING!!! sebelum menggunakan program ini sesuaikan terlebih dahulu path UPLOAD_FOLDER
UPLOAD_FOLDER = 'D:/KU LI AH/SEMESTER 6/Pengolahan & Analisis Citra Digital/projek/140810200012_Proyek PACD/static/uploads'
ALLOWED_EXTENSIONS = set(['png', 'jpg', 'jpeg', 'PNG'])

app = Flask(__name__)
app.config['SEND_FILE_MAX_AGE_DEFAULT'] = 0
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.secret_key = "secret key"

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1] in ALLOWED_EXTENSIONS

def process_image(filename):
    rgbImage = cv2.imread(filename)
    fig = plt.figure(figsize=(113, 60))

    # Resizing image
    img = imutils.resize(rgbImage, height=200)
    flat_img = np.reshape(img, (-1, 3))

    # Perform k-means clustering
    kmeans = KMeans( random_state=0)
    kmeans.fit(flat_img)
    k = kmeans.n_clusters  # Mengambil jumlah cluster dari hasil clustering

    if k > 0:
        # Calculate color proportions
        percentages = (np.unique(kmeans.labels_, return_counts=True)[1]) / flat_img.shape[0]
        p_and_c = list(zip(percentages, kmeans.cluster_centers_))
        p_and_c = sorted(p_and_c, key=lambda x: x[0], reverse=True)

        # Filter out duplicate colors
        unique_colors = []
        unique_p_and_c = []
        for p, c in p_and_c:
            hex_color = mcolors.to_hex(c / 255.0)
            if hex_color not in unique_colors:
                unique_colors.append(hex_color)
                unique_p_and_c.append((p, c))

        # Create GridSpec from matplotlib to show the results
        gs = GridSpec(2, 2, figure=fig)

        oriImage = fig.add_subplot(gs[0, 0])
        clusImage = fig.add_subplot(gs[0, 1])
        colorProportions = fig.add_subplot(gs[1, :])

        # Display the original image
        oriImage.imshow(cv2.cvtColor(rgbImage, cv2.COLOR_BGR2RGB))
        oriImage.set_title('Original Image', fontsize=150)
        oriImage.axis('off')

        # Display the clustered image
        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)
        _, labels, centers = cv2.kmeans(flat_img.astype(np.float32), k, None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS)
        centers = centers.astype(np.uint8)
        clusteredImage = centers[labels.flatten()].reshape(img.shape)
        clusImage.imshow(cv2.cvtColor(clusteredImage, cv2.COLOR_BGR2RGB))
        clusImage.set_title('Clustered Image', fontsize=130)
        clusImage.axis('off')

        # Plot color proportions as a pie chart
        proportions = [round(p * 100, 2) for p, _ in unique_p_and_c]
        colors = [mcolors.to_hex(color / 255.0) for _, color in unique_p_and_c]
        labels = [f'{p}%\n{c}' for p, c in zip(proportions, colors)]

        colorProportions.pie(proportions, labels=labels, colors=colors, startangle=90, counterclock=False
                             , pctdistance=0.85, textprops={'fontsize': 0})

        # Set aspect ratio of the pie chart to be equal so that it becomes a circle
        colorProportions.axis('equal')
        colorProportions.set_title('Proportions of colors in the image', fontsize=150)
        colorProportions.legend(labels, loc='center', bbox_to_anchor=(0.5, -0.1), ncol=len(labels), fontsize='100')

        # Simpan hasil gambar
        hasil = secure_filename(filename.split('.')[0]) + "_proses.jpg"
        pathresult = os.path.join(app.config['UPLOAD_FOLDER'], hasil)
        plt.savefig(pathresult)  # Save the result image
        plt.close(fig)

        plt.tight_layout()  # Adjust spacing between subplots

        return hasil
    else:
        print("No clusters found in the image.")
        abort(400, "No clusters found in the image.")

@app.route('/')
def home():
    return render_template('home.html')

@app.route('/process', methods=['POST'])
def process():
    file = request.files['file']
    if file and allowed_file(file.filename):
        filename = secure_filename(file.filename)
        file.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))
        img_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)  # Full path of the uploaded image
        try:
            processed_img_name = process_image(img_path)  # Process the image and get the processed image name
            return render_template('home.html', process_img_name=processed_img_name)
        except Exception as e:
            error_message = f"Error processing the image: {str(e)}"
            return render_template('home.html', error_message=error_message)
    else:
        error_message = "Invalid file extension. Only PNG and JPG files are allowed."
        return render_template('home.html', error_message=error_message)

if __name__ == '__main__':
    app.run(debug=True)
