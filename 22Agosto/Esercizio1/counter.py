# Streamlit application implementing a counter that can be decremented or incremented through buttons
import streamlit as st

# Create a counter displayed in a container
counter_container = st.container()
with counter_container:
    # set initial counter value
    if "counter" not in st.session_state:
        st.session_state.counter = 0
    st.write("Counter:", st.session_state.counter)

def click_plus_button():
    # Increment the counter
    st.session_state.counter += 1

def click_minus_button():
    # Decrement the counter
    st.session_state.counter -= 1

# Create a + button displayed as +
st.button('Plus', on_click=click_plus_button)
# Create a - button displayed as -
st.button('Minus', on_click=click_minus_button)

# Create a sidebar to insert the name of the user and print, after the submission, Hello User!
with st.sidebar.form(key='user_form'):
    user_name = st.text_input("Enter your name:")
    submit_button = st.form_submit_button(label='Submit')
    if submit_button:
        st.write(f"Hello {user_name}!")
