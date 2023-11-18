import pandas
import streamlit as st

from models.buying_price import BuyingPriceMarket, BuyingPriceProduct

# datas
markets = pandas.DataFrame(BuyingPriceMarket.get_all())

# filter
market = st.sidebar.selectbox("Choose a market : ", markets["market_name"])

products_name = pandas.DataFrame(
    BuyingPriceProduct.get_products_name_for_market(market)
)
product = st.sidebar.selectbox(
    "Choose a product : ", products_name["product_name"]
)
year = st.sidebar.selectbox("Choose a date : ", ["2023", "2022", "2021"])
'For the market "', market, '"'
'Prices for : "', product, '", in : ', year

prices = pandas.DataFrame(
    BuyingPriceProduct.get_product_for_market(market, product)
)

# convert
prices["avg_price"] = prices["avg_price"].astype(float)
prices["date_price"] = pandas.to_datetime(prices["date_price"])
prices["date_price"] = prices["date_price"].dt.strftime("%d/%m/%Y")

# filtering
prices = prices[prices["date_price"].str.contains(year)]
prices = prices[prices["product_name"].str.contains(product)]
# display
tab1, tab2, tab3, tab4 = st.tabs(["Table", "Line", "Bar", "Area"])

with tab1:
    st.dataframe(prices.style.highlight_min(subset="avg_price"))
with tab2:
    st.line_chart(prices, y="avg_price", x="date_price")
with tab3:
    st.bar_chart(prices, y="avg_price", x="date_price")
with tab4:
    st.area_chart(prices, y="avg_price", x="date_price")
