# -*- coding: utf-8 -*-
"""
Created on Wed Mar 29 08:51:41 2023

@author: michele and Antoine
"""
from collections import OrderedDict

import streamlit as st
import config
from tabs import intro, dataset_tab, dataviz_tab, modeling_tab, recommendation_demo_tab, conclusion_tab
import os

current_file_path = os.path.realpath(__file__) 
current_dir = os.path.dirname(current_file_path)
assets_dir = os.path.join(current_dir, '..',  'assets')
page_icon_popcorn = os.path.join(assets_dir, "linkedin-logo.png")


st.set_page_config(
    page_title=config.TITLE,
    page_icon="https://datascientest.com/wp-content/uploads/2020/03/cropped-favicon-datascientest-1-32x32.png",
)

with open("style.css", "r") as f:
    style = f.read()

st.markdown(f"<style>{style}</style>", unsafe_allow_html=True)

TABS = OrderedDict(
    [
        (intro.sidebar_name, intro),
        (dataset_tab.sidebar_name, dataset_tab),
        (dataviz_tab.sidebar_name, dataviz_tab),
        (modeling_tab.sidebar_name, modeling_tab),
        (recommendation_demo_tab.sidebar_name, recommendation_demo_tab),
        (conclusion_tab.sidebar_name, conclusion_tab),
    ]
)


def run():
    

    tab_name = st.sidebar.radio("", list(TABS.keys()), 0)
    st.sidebar.markdown("---")
    st.sidebar.markdown(f"## {config.PROMOTION}")

    st.sidebar.markdown("### Team members:")
    for member in config.TEAM_MEMBERS:
        st.sidebar.markdown(member.sidebar_markdown(), unsafe_allow_html=True)

    tab = TABS[tab_name]

    tab.run()


if __name__ == "__main__":
    run()