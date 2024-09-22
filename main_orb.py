import cv2

# File paths
small_image_path = '/objects/sailboat.png'
large_image_path = '/images/kotor_small.png'

# Load the images
small_image = cv2.imread(small_image_path)
large_image = cv2.imread(large_image_path)

# Convert images to grayscale for better matching performance
small_image_gray = cv2.cvtColor(small_image, cv2.COLOR_BGR2GRAY)
large_image_gray = cv2.cvtColor(large_image, cv2.COLOR_BGR2GRAY)

# Initialize the ORB detector
orb = cv2.ORB_create()

# Find keypoints and descriptors with ORB for both images
kp1, des1 = orb.detectAndCompute(small_image_gray, None)
kp2, des2 = orb.detectAndCompute(large_image_gray, None)

# Create BFMatcher object
bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)

# Match descriptors
matches = bf.match(des1, des2)

# Sort them in the order of their distance (lower distance is better)
matches = sorted(matches, key=lambda x: x.distance)

# Set a threshold for match distance
distance_threshold = 60  # Adjust this value as needed (lower value for stricter matching)

# Filter out the matches with distance greater than the threshold
good_matches = [m for m in matches if m.distance < distance_threshold]

# Draw only the good matches
img_matches = cv2.drawMatches(small_image, kp1, large_image, kp2, good_matches, None, flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)

# Display the image with OpenCV
cv2.imshow('Good Matches (Filtered by Threshold)', img_matches)

# Wait for a key press and close the image window
cv2.waitKey(0)
cv2.destroyAllWindows()