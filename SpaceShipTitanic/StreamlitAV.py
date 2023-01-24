import streamlit as st
import streamlit.components.v1 as components
import base64


def sidebar(name):
    st.sidebar.header(name)


def write_h1(text):
    st.write(f"# {text}")


def write_h2(text):
    st.write(f"## {text}")


def write_h3(text):
    st.write(f"### {text}")


def write_h4(text):
    st.write(f"#### {text}")


def write_h5(text):
    st.write(f"##### {text}")


def selectbox(headline, options):
    option = st.selectbox(
        headline,
        options=options
    )
    return option


def selected_filter_values(filter_name, data_frame):
    selected_values = st.sidebar.multiselect(filter_name, list(data_frame.columns), list(data_frame.columns))
    return selected_values


def multiselct(headline, all_select_options, already_selected=[]):
    options = st.multiselect(
        headline,
        all_select_options,
        already_selected)
    return options


def radio(haadline, selections):
    option = st.radio(
        haadline,
        selections)
    return option


def filter_func(data_frame, table_name, selected_values):
    df_selected_data = data_frame[selected_values]
    st.header(table_name)
    st.dataframe(df_selected_data)


def file_download(dataframe, file_name):
    csv = dataframe.to_csv(index=False)
    b64 = base64.b64encode(csv.encode()).decode()
    href = f'<a href="data:file/csv;base64,{b64}" download="{file_name}.csv">Download CSV File</a>'
    return href


def markdown(data_frame, file_name):
    st.markdown(file_download(data_frame, file_name), unsafe_allow_html=True)


def display_dataframe(data_frame, table_name):
    st.header(table_name)
    st.dataframe(data_frame)


