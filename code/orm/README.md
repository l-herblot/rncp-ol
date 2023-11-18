# Create object for models

## Create a BuyingPriceMarket object
market = BuyingPriceMarket(name='Market Name', key='Market Key', url='Market URL')

## Create a BuyingPriceMarketStep object
market_step = BuyingPriceMarketStep(name='Market Step Name', key='Market Step Key')

## Create a BuyingPriceMarketMarketStep object
market_market_step = BuyingPriceMarketMarketStep(id_market=1, id_market_type=2)

## Create a BuyingPriceProduct object
product = BuyingPriceProduct(
    id_market=1,
    id_market_type=2,
    date_price='2023-07-28',
    name='Product Name',
    avg_price=10.50,
    min_price=9.75,
    max_price=12.25,
    unit='kg'
)
