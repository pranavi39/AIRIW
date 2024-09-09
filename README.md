# Pawfectly Choosen - Pet Shop Search App

### Overview
Pawfectly Choosen is a Streamlit-based web application designed to help pet owners find the perfect products for their pets. The app uses TF-IDF (Term Frequency-Inverse Document Frequency) and cosine similarity for efficient product search and retrieval, offering personalized recommendations based on user queries and pet selection.

### Features
* Pet-Based Search: Users can select a pet (Dog, Cat, Fish) and search for products relevant to that pet, such as food, toys, and accessories.
* TF-IDF & Cosine Similarity: Utilizes machine learning techniques for text-based product description search, ensuring accurate and relevant results.
* User Authentication: Users can log in to save products to a wishlist for easy access later.
* Wishlist: Logged-in users can add desired products to their wishlist and view it in the sidebar.
* Customizable UI: The app features a dynamic background and clean interface for an engaging user experience.

### How It Works
* Users select their pet type from a sidebar dropdown (Dog, Cat, Fish).
* They enter a search term (e.g., "food," "toy," "aquarium") in the search bar.
* The app retrieves the most relevant products using TF-IDF and cosine similarity and displays the results.
* Logged-in users can add products to their wishlist for future reference.

### Technologies Used
* Streamlit for the web interface
* Pandas for data handling
* TF-IDF and Cosine Similarity for information retrieval (via scikit-learn)
* HTML & CSS for background styling and user interface customization

### Data
* products.csv: Contains product information, including description and pet type.
* users.csv: Contains user credentials for login functionality.
