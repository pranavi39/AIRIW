import streamlit as st
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Load the data from the CSV file
@st.cache_data
def load_data():
    # Load the CSV file
    df = pd.read_csv("products.csv")
    return df

# Load the user data from the CSV file
@st.cache_data
def load_users():
    # Load the CSV file
    users_df = pd.read_csv("users.csv")
    return users_df

# Calculate tf-idf scores for product descriptions
def calculate_tfidf(data):
    # Initialize the TfidfVectorizer
    vectorizer = TfidfVectorizer()
    # Fit and transform the product descriptions
    tfidf_matrix = vectorizer.fit_transform(data['Description'])
    return tfidf_matrix, vectorizer

# Calculate cosine similarity using VSM
def calculate_vsm_similarity(search_vector, tfidf_matrix):
    return cosine_similarity(search_vector, tfidf_matrix).flatten()

# Function to set background image using HTML and CSS
def set_background(bg_image_url):
    """
    Function to set background image using HTML and CSS
    """
    # Set background image using HTML and CSS
    st.markdown(
        f"""
        <style>
            .container {{
                background: url("{bg_image_url}") no-repeat center center fixed;
                background-size: contain;
                padding: 100px;
            }}
            .header {{
                font-size: 36px;
                font-weight: bold;
                color: white;
                text-align: center;
                margin-bottom: 30px;
            }}
        </style>
        """,
        unsafe_allow_html=True
    )

# Streamlit UI
def main():
    # Set page title
    st.title("Pawfectly Choosen")

    # Load the data
    data = load_data()

    # Load the user data
    users_df = load_users()

    # Calculate tf-idf scores for the data
    tfidf_matrix, vectorizer = calculate_tfidf(data)

    # Sidebar options
    selected_pet = st.sidebar.selectbox("Choose the pet:", ('Dog', 'Cat', 'Fish'))
    st.sidebar.write(f"You selected: {selected_pet}")

    # Wishlist sidebar section
    st.sidebar.markdown("---")
    #st.sidebar.header("Wishlist")
    if 'wishlist_df' not in st.session_state:
        st.session_state.wishlist_df = []  # Initialize wishlist as a list

    # Main content
    set_background('https://imgur.com/PUGOOuU.jpg')  # Set background image
    st.markdown('<div class="container">', unsafe_allow_html=True)  # Open container
    st.markdown('<div class="header">Welcome to Pawfectly Choosen! Your one-stop pet shop</div>', unsafe_allow_html=True)

    # Initialize logged_in attribute if it doesn't exist
    if 'logged_in' not in st.session_state:
        st.session_state['logged_in'] = False

    # Login section
    if not st.session_state.logged_in:
        st.subheader("Login to Add Products to Wishlist")
        username = st.text_input("Username")
        password = st.text_input("Password", type="password")
        login_button = st.button("Login")

        if login_button:
            # Check if username and password are in the users dataframe
            if any((users_df['username'] == username) & (users_df['password'] == password)):
                st.session_state.logged_in = True
                st.success("Login successful! You can now add products to your wishlist.")
            else:
                st.error("Invalid username or password. Please try again.")

    # Search bar for products
    search_term = st.text_input("Search for products (e.g., food, toy, aquarium):")

    if search_term:
        # Filter the data based on the selected pet
        filtered_data = data[data['Pet'] == selected_pet]

        # Transform the search term using the same vectorizer
        search_vector = vectorizer.transform([search_term])

        # Calculate similarity scores using VSM
        similarity_scores = calculate_vsm_similarity(search_vector, tfidf_matrix[filtered_data.index])

        # Create a DataFrame for the results
        results_df = pd.DataFrame({'index': filtered_data.index, 'score': similarity_scores})

        # Filter the data based on the similarity score and search term
        search_results = pd.concat([filtered_data, results_df.set_index('index')], axis=1)
        search_results = search_results[search_results['Description'].str.contains(search_term, case=False)]
        search_results = search_results.sort_values(by='score', ascending=False)

        # Display the search results
        if not search_results.empty:
            st.header("Search Results")
            for index, row in search_results.iterrows():
                st.write(f"**Product:** {row['Product']}")
                st.write(f"**Price:** {row['Price']}")
                st.write(f"**Description:** {row['Description']}")
                # Add to Wishlist button
                if st.session_state.logged_in:
                    wishlist_button = st.button(f"Add {row['Product']} to Wishlist")
                    if wishlist_button:
                        wishlist_data = {'Product': row['Product'], 'Price': row['Price'], 'Description': row['Description']}
                        st.session_state.wishlist_df.append(wishlist_data)  # Append to wishlist
        else:
            st.write("No results found for your search.")

    # Display wishlist in the sidebar
    if st.session_state.logged_in:
        st.sidebar.header("Your Wishlist")
        for item in st.session_state.wishlist_df:
            st.sidebar.write(f"**Product:** {item['Product']}")
            st.sidebar.write(f"**Price:** {item['Price']}")
            st.sidebar.write(f"**Description:** {item['Description']}")

    st.markdown('</div>', unsafe_allow_html=True)  # Close container

if __name__ == "__main__":
    main()
