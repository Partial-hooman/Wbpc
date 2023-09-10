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
csv_data = csv_data.applymap(str.lower)
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
te_ary = te.fit(data_list).transform(data_list)


# Initialize min_support and min_confidence as global variables with default values
#min_support = 0.01
#min_confidence = 0.01
if 'min_support' not in st.session_state:
   st.session_state['min_support'] = 0.05
if 'min_confidence' not in st.session_state:
   st.session_state['min_confidence'] = 0.2

# Function to generate association rules
def generate_association_rules(data, min_support, min_confidence,dat):
    # Convert the data to one-hot encoded format

    #st.write(min_support)
    #st.write(min_confidence)
    #one_hot = pd.get_dummies(sum(data, []), prefix='', prefix_sep='')
    dat = dat
    dat.drop(['=======','nan'],axis=1,inplace=True)
    # Use Apriori algorithm to find frequent item sets
    #st.write(dat)
    frequent_item_sets = apriori(dat, min_support=min_support, use_colnames=True)
    #st.write(frequent_item_sets)
    # Generate association rules
    rules = association_rules(frequent_item_sets, metric="confidence", min_threshold=min_confidence)
    #st.write(rules)
    return rules

# Function to generate recommendations based on shopping list
def generate_recommendations(shopping_list, data, min_support, min_confidence):
    # Create a DataFrame with the shopping list
    shopping_df = pd.DataFrame({'items': shopping_list})
    #st.write(shopping_df)
    # Convert the data to one-hot encoded format
    #one_hot = pd.get_dummies(data, prefix='', prefix_seps='')
    st.write(len(te.columns_))
    dat = pd.DataFrame(te_ary, columns=te.columns_)
    dat.drop(['=======','nan'],axis=1,inplace=True)
    
    # Add columns for items in the shopping list (set to 0 initially)
    #for item in shopping_list:
    #    dat[item] = 0
    #st.write(min_support)
    #st.write(min_confidence)
    # Apply Apriori algorithm to find recommendations
    frequent_item_sets = apriori(dat, min_support=min_support, use_colnames=True)
    rules = association_rules(frequent_item_sets, metric="confidence", min_threshold=min_confidence)

    # Filter rules based on shopping list
    recommended_items = []
    for index, row in shopping_df.iterrows():
        item = row['items']
        relevant_rules = rules[rules['antecedents'].apply(lambda x: item in x)]
        recommended_items.extend(relevant_rules['consequents'].explode().unique())

    # Remove items already in the shopping list
    recommended_items = [item for item in recommended_items if item not in shopping_list]
    #st.write(recommended_items)
    return recommended_items[:5]  # Return the top 5 recommendations

# Streamlit app
def main():
    #global min_support, min_confidence  # Declare min_support and min_confidence as global

    st.title("Shopping List Recommendation App")
    #st.write(csv_data)
    #st.write(data_list[0])
    # Page selection
    page = st.sidebar.selectbox("Select a page", ["Welcome", "Shopping List", "Options", "Results"])

    if page == "Welcome":
        st.header("Welcome to the Shopping List Recommendation App")
        st.write("Please select a page from the sidebar.")

    elif page == "Shopping List":
        st.header("Shopping List")
        min_support = st.session_state.min_support
        min_confidence = st.session_state.min_confidence
        dat = pd.DataFrame(te_ary, columns=te.columns_)
        top_items = dat.melt().value_counts().reset_index()
        top_items.drop(['value'], axis=1, inplace=True)
        list1 = ['=======','nan']
        top_items = top_items[top_items.variable.isin(list1) == False]
        #st.write(top_items)
        top_items.columns = ['Item', 'Count']
        top_items['Percentage'] = (top_items['Count'] / len(csv_data)) * 100
        if 'shopping_list' not in st.session_state:
            st.session_state['shopping_list'] = []
        
        # Create an empty shopping list
        #shopping_list = st.text_input("Add items to your shopping list (comma-separated)")
        #st.write(csv_data)
        #st.write(te_ary)
        #st.write(data_list)
        # Create a dropdown with items from the database
        item_list = te.columns_ # Assuming the columns are item names
        #st.write(item_list)
        
        del item_list[0]
        item_list = [j for i,j in enumerate(item_list) if j!="nan"]
        selected_item = st.multiselect("Select an item from the database", item_list)
        if st.button("clear shopping list"):
           st.session_state.shopping_list = []
        
         
        if st.button("Add to Shopping List"):
            #st.session_state.shopping_list = [item.strip().lower() for item in shopping_list.split(',')]
            #if selected_item:
            #st.write(selected_item)
            #st.write(type (selected_item))
            #itemstoadd = sum(selected_item, [])
           for i in selected_item:
               if i not  in st.session_state.shopping_list:
                  st.session_state.shopping_list.append(i)
            
           st.success("Items added to your shopping list: {}".format(st.session_state.shopping_list))
           
        st.write("items present in shopping list:")
        st.dataframe(st.session_state.shopping_list)
        rc = []
        try:
          rc = generate_recommendations(st.session_state.shopping_list, data_list, min_support, min_confidence,dat)
          st.write("recommended_items:")
          if len(rc) > 0:
           st.write(rc)
          else:
           rc = list(top_items.head(5)["Item"])
           st.write(list(top_items.head(5)["Item"]))
        except Exception as e:
           st.write("recommended_items:")
           rc = list(top_items.head(5)["Item"]) 
           st.write(list(top_items.head(5)["Item"]))
           st.write(e)
    elif page == "Options":
        st.header("Options")

        # Support and Confidence thresholds
        min_support = st.slider("Minimum Support", 0.01, 1.0, st.session_state.min_support)
        min_confidence = st.slider("Minimum Confidence", 0.01, 1.0, st.session_state.min_confidence)
        st.session_state.min_support = min_support
        st.session_state.min_confidence = min_confidence
        st.write("You can adjust the minimum support and confidence thresholds here.")

    elif page == "Results":
        min_support = st.session_state.min_support
        min_confidence = st.session_state.min_confidence
        st.header("Results")
        #st.write(te_ary)
        # Generate association rules based on user-defined options
        rules = generate_association_rules(data_list, min_support, min_confidence)
        if rules.empty:
            st.warning("No association rules found with the given thresholds. Try lowering the thresholds.")
        else:
            # Display association rules
            st.subheader("Association Rules")
            st.write(rules)

        # Display top 20 items with percentages
        st.subheader("Top 20 Items with Percentages")
        dat = pd.DataFrame(te_ary, columns=te.columns_)
        top_items = dat.melt().value_counts().reset_index()
        top_items.drop(['value'], axis=1, inplace=True)
        list1 = ['=======','nan']
        top_items = top_items[top_items.variable.isin(list1) == False]
        #st.write(top_items)
        top_items.columns = ['Item', 'Count']
        top_items['Percentage'] = (top_items['Count'] / len(csv_data)) * 100
        st.write(top_items.head(20))
        #rc = generate_recommendations(st.session_state.shopping_list, data_list, min_support, min_confidence)
        #st.write("recommended_items:")
        #if len(rc) > 0:
        #  st.write(rc)
        #else:
        #  st.write(top_items.head(5))
        #st.write(generate_recommendations(st.session_state.shopping_list, data_list, min_support, min_confidence))
        # Create a scatter plot of support vs. confidence
        plt.figure(figsize=(8, 6))
        sns.scatterplot(x='support', y='confidence', data=rules)
        plt.title('Association Rules - Support vs. Confidence')
        st.pyplot(plt)

if __name__ == "__main__":
    main()
