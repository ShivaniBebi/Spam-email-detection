import streamlit as st
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score

st.set_page_config(page_title="Spam Email Detector", page_icon="📧")

# Title
st.title("📧 Spam Email Detection App")
st.write("This app classifies emails as **Spam** or **Not Spam** using Machine Learning (Naive Bayes).")

# Step 1: Load and label data
@st.cache_data
def load_data():
    df = pd.read_csv("emails.csv", encoding='latin1', nrows=10000)

    if 'message' not in df.columns:
        st.error("❌ The dataset must have a 'message' column.")
        return None

    # Rule-based labeling function
    def label_spam(msg):
        spam_keywords = ['free', 'win', 'click here', 'buy now', 'urgent', 'prize', 'congratulations']
        return 1 if any(word in msg.lower() for word in spam_keywords) else 0

    df['label'] = df['message'].apply(label_spam)
    df.rename(columns={'message': 'text'}, inplace=True)
    return df

df = load_data()

if df is not None:
    # Step 2: Train the modelqw
    X = df['text']
    y = df['label']

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    vectorizer = CountVectorizer(stop_words='english')
    X_train_vec = vectorizer.fit_transform(X_train)
    X_test_vec = vectorizer.transform(X_test)

    model = MultinomialNB()
    model.fit(X_train_vec, y_train)

    # Step 3: Evaluate
    y_pred = model.predict(X_test_vec)
    acc = accuracy_score(y_test, y_pred)
    st.info(f"📊 Model Accuracy: **{acc:.2%}** on test data")

    # Step 4: User input
    st.subheader("🔍 Try it yourself")
    user_input = st.text_area("Enter a message to test:", height=100)

    if st.button("Check for Spam"):
        if user_input.strip() == "":
            st.warning("Please enter a message first.")
        else:
            user_vec = vectorizer.transform([user_input])
            prediction = model.predict(user_vec)[0]
            result = "🚫 Spam" if prediction == 1 else "✅ Not Spam"
            st.success(f"**Prediction:** {result}")

