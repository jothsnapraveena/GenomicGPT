import streamlit as st
import pandas as pd
from agenticAI import *
from list_constants import *

# Set page configuration for full width
st.set_page_config(layout="wide")

def load_suggestions():
    """Load suggestions from a file (mock implementation)."""
    return {
        "genes": genes,
        "diseases": diseases
    }
def get_complementary_color(hex_color):
    hex_color = hex_color.lstrip('#')
    r = int(hex_color[0:2], 16)
    g = int(hex_color[2:4], 16)
    b = int(hex_color[4:6], 16)
    r_complement = 255 - r
    g_complement = 255 - g
    b_complement = 255 - b
    complementary_hex = f"#{r_complement:02x}{g_complement:02x}{b_complement:02x}"
    
    print("get_complementary_color", hex_color, complementary_hex)
    return complementary_hex

def map_confidence_to_category(confidence_value):
    confidence_value = int(confidence_value)
    print(confidence_value, type(confidence_value))
    """Map confidence value (0-10) to a category."""
    if confidence_value < 3:
        return 'Refuted'
    elif confidence_value < 5:
        return 'Moderate'
    elif confidence_value < 6.5:
        return 'Limited'
    elif confidence_value < 8:
        return 'Strong'
    elif confidence_value <= 9.5:
        return 'Definitive'
    else:
        return 'Unknown'

def main():
    # Custom CSS to center-align the title and reduce space
    color1 = '#e4c5a9' # #c49a6e #5e3825 #675e58 #bc764f #e4c5a9
    color2 = '#d0beb9'
    tcol0, tcol1, tcol2 = st.columns([5, 5, 5])
    with tcol1:
        st.image("logo.jpeg", width=1000)
    suggestions = load_suggestions()
    
    col1, col2 = st.columns([1, 2])
    
    with col1:
        gene = st.selectbox("Gene Name", options=[""] + suggestions['genes'])
        disease = st.selectbox("Disease Name", options=[""] + suggestions['diseases'])
        #submit = st.button("Submit")
        if gene != None and gene != '':
            df_done = pd.read_csv("G2P_Merged_Data.csv")
            filtered_df = df_done[df_done['gene symbol'] == gene]
            #print(filtered_df)
            table_data = "<table border='1' style='width:100%; border-collapse: collapse;'>"
            table_data += "<tr style='background-color: " + color1 + ";text-align:center'><th colspan='3'>" + gene + " - Disease Pairs</th></tr>"
            table_data += "<tr style='background-color: " + color1 + ";text-align:center'><th>Disease Name</th>"
            table_data += "<th>Molecular Mechanism</th>"
            table_data += "<th>Confidence</th></tr>"
            for i in range(len(filtered_df)):
                rows = filtered_df.iloc[i]
                #print("rows", rows)
                table_data += f"<tr><td>{rows['disease name']}</td>"
                table_data += f"<td>{rows['molecular mechanism']}</td>"
                table_data += f"<td>{rows['confidence']}</td></tr>"
            table_data += '</table>'
            st.write(table_data, unsafe_allow_html=True)
    with col2:
        if gene != None and gene != '' and disease != None and disease != '':
            st.header(gene + " - " + disease)
            if not gene.strip():
                st.error("Gene Name is required!")
            elif not disease.strip():
                st.error("Disease Name is required!")
            else:
                final_report, print_outs_final = autonomous_mechanism_discovery(gene, disease)
                #final_report = json.load(final_report)
                #print("final_report", final_report)
                #print("print_outs_final", print_outs_final)
                #print("final_report['gene']", final_report['gene'])
                table_html = "<table border='1' style='width:100%; border-collapse: collapse;'>"
                # #c49a6e #5e3825 #675e58 #bc764f #e4c5a9
                table_html += "<tr style='background-color: " + color1 + ";text-align:center'><th>Gene</th>"
                table_html += "<th>Mechanism of Disease</th>"
                table_html += "<th>Disease Name</th>"
                table_html += "<th>Evidence Confidence</th>"
                table_html += "<th>Confidence Score</th></tr>"
                table_html += f"<tr><td>{final_report['gene']}</td>"
                table_html += f"<td>{final_report['mechanism']}</td>"
                table_html += f"<td>{final_report['disease']}</td>"

                val_ret = str(map_confidence_to_category(final_report['confidence_score']))
                table_html += f"<td>{val_ret}</td>"
                table_html += f"<td align='right'>{final_report['confidence_score']}</td></tr>"
                
                for i in range(len(print_outs_final)):
                    table_html += "<tr><td colspan='5' style='margin:0px 0px 0px 0px;padding:0px 0px 0px 0px;border:0px'>"
                    table_html += "<table style='margin:0px 0px 0px 0px; border-collapse: collapse; width: 100%; border:0px'><tr>"
                    colors = ['#d0beb9', '#c2b1b0', '#b5a49f', '#a79796', '#998885', '#8a7c74', '#7c6f67', '#675e58']
                    
                    #if i < 6:
                    #    table_html += "<th style='background-color:" + colors[i] + ";color:" + get_complementary_color(colors[i]) + "'>"
                    #else:
                    #    table_html += "<th style='background-color: #675e58;text-color:" + get_complementary_color('#675e58') + "'>"
                    
                    table_html += "<th style='background-color: " + color2 + ";'>"
                    table_html += print_outs_final[i][0]
                    table_html += "</th></tr>"

                    for j in range(1, len(print_outs_final[i])):
                        if 'Evidence ' not in print_outs_final[i][j][:20]:
                            table_html += "<tr><td>"
                            table_html += print_outs_final[i][j]
                            table_html += "</td></tr>"
                    #table_html += str(review_articles)
                    table_html += "</table>"
                    table_html += "</td></tr>"
                table_html += str(final_report['justification'])

                table_html += "</table>"
                
                st.write(table_html, unsafe_allow_html=True)

if __name__ == "__main__":
    main()
