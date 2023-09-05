import streamlit as st
import pandas as pd
from mlxtend.frequent_patterns import apriori
from mlxtend.frequent_patterns import association_rules
import matplotlib.pyplot as plt
import seaborn as sns
import mlxtend.frequent_patterns as fp
import mlxtend.preprocessing as preproc
from mlxtend.preprocessing import TransactionEncoder
# Sample CSV data (replace with your own dataset)
 # Replace 'dataSort.csv' with your file path
csv_data = pd.read_csv('dataSort.csv', low_memory=False)
csv_data.drop(columns="Unnamed: 0", inplace=True)
csv_data=csv_data.astype(str)
#csv_data.drop(index=csv_data.index[0], axis=1, inplace=True)
print(csv_data)
#data = (csv_data.to_csv()).split('\n')
data_list = csv_data.to_numpy().tolist()
#for i in data:
 #ita = i.split(',')
 #del ita[0]
 #data_list.append(i.split(','))
#del data_list[0]

te = TransactionEncoder()

# Transform the data into a one-hot encoded DataFrame
#te_ary = te.fit_transform(csv_data)
#te_ary = te.fit(data_list).transform(data_list)


# Initialize min_support and min_confidence as global variables with default values
min_support = 0.1
min_confidence = 0.5

# Function to generate association rules
def generate_association_rules(data, min_support, min_confidence):
    # Convert the data to one-hot encoded format



    one_hot = pd.get_dummies(data, prefix='', prefix_sep='')

    # Use Apriori algorithm to find frequent item sets
    frequent_item_sets = apriori(one_hot, min_support=min_support, use_colnames=True)

    # Generate association rules
    rules = association_rules(frequent_item_sets, metric="confidence", min_threshold=min_confidence)

    return rules

# Function to generate recommendations based on shopping list
def generate_recommendations(shopping_list, data, min_support, min_confidence):
    # Create a DataFrame with the shopping list
    shopping_df = pd.DataFrame({'items': shopping_list})

    # Convert the data to one-hot encoded format
    one_hot = pd.get_dummies(data, prefix='', prefix_sep='')

    # Add columns for items in the shopping list (set to 0 initially)
    for item in shopping_list:
        one_hot[item] = 0

    # Apply Apriori algorithm to find recommendations
    frequent_item_sets = apriori(one_hot, min_support=min_support, use_colnames=True)
    rules = association_rules(frequent_item_sets, metric="confidence", min_threshold=min_confidence)

    # Filter rules based on shopping list
    recommended_items = []
    for index, row in shopping_df.iterrows():
        item = row['items']
        relevant_rules = rules[rules['antecedents'].apply(lambda x: item in x)]
        recommended_items.extend(relevant_rules['consequents'].explode().unique())

    # Remove items already in the shopping list
    recommended_items = [item for item in recommended_items if item not in shopping_list]

    return recommended_items[:5]  # Return the top 5 recommendations

# Streamlit app
def main():
    global min_support, min_confidence  # Declare min_support and min_confidence as global

    st.title("Shopping List Recommendation App")
    st.write(csv_data)
    st.write(data_list[0])
    # Page selection
    page = st.sidebar.selectbox("Select a page", ["Welcome", "Shopping List", "Options", "Results"])

    if page == "Welcome":
        st.header("Welcome to the Shopping List Recommendation App")
        st.write("Please select a page from the sidebar.")

    elif page == "Shopping List":
        st.header("Shopping List")

        # Create an empty shopping list
        shopping_list = st.text_input("Add items to your shopping list (comma-separated)")
        #st.write(csv_data)
        st.write(te_ary)
        st.write(data_list)
        # Create a dropdown with items from the database
        item_list = te.columns_  # Assuming the columns are item names
        st.write(item_list)
        selected_item = st.selectbox("Select an item from the database", item_list)

        if st.button("Add to Shopping List"):
            shopping_list = [item.strip() for item in shopping_list.split(',')]
            if selected_item:
                shopping_list.append(selected_item)
            st.success("Items added to your shopping list: {}".format(shopping_list))

    elif page == "Options":
        st.header("Options")

        # Support and Confidence thresholds
        min_support = st.slider("Minimum Support", 0.0, 1.0, 0.01)
        min_confidence = st.slider("Minimum Confidence", 0.0, 1.0, 0.01)

        st.write("You can adjust the minimum support and confidence thresholds here.")

    elif page == "Results":
        st.header("Results")

        # Generate association rules based on user-defined options
        rules = generate_association_rules(te_ary, min_support, min_confidence)
        if rules.empty:
            st.warning("No association rules found with the given thresholds. Try lowering the thresholds.")
        else:
            # Display association rules
            st.subheader("Association Rules")
            st.write(rules)

        # Display top 20 items with percentages
        st.subheader("Top 20 Items with Percentages")
        top_items = te_ary.melt().value_counts().reset_index()
        top_items.columns = ['Item', 'Count']
        top_items['Percentage'] = (top_items['Count'] / len(csv_data)) * 100
        st.write(top_items.head(20))

        # Create a scatter plot of support vs. confidence
        plt.figure(figsize=(8, 6))
        sns.scatterplot(x='support', y='confidence', data=rules)
        plt.title('Association Rules - Support vs. Confidence')
        st.pyplot(plt)

if __name__ == "__main__":
    main()
