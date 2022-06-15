# API Key
FRED_API_KEY = '4a43b9573a9c8df061ad6e183ef07110'

# Time Series data code
# Employment
TOTAL_NONFARM = 'PAYEMS'

# Monetary Policy
FED_FUNDS_RATE_DAILY = 'DFF'
FED_FUNDS_RATE_MONTHLY = 'FEDFUNDS'

# Inflation - CPI data has lots of version
# Seasonally Adjusted
CPI_US_ADJUSTED = {'ALL_URBAN': 'CPIAUCSL',
                   'ALL_URBAN_LESS_FOOD_ENERGY': 'CPILFESL'
                   }
# Not Seasonally Adjusted
CPI_US_UNADJUSTED = {'ALL_US': 'CPALTT01USM657N',
                     }

# Bond Market
# US Treasury yield curve - daily
TREASURY_DAILY = {'1M': 'DGS1MO',
                  '3M': 'DGS3MO',
                  '6M': 'DGS6MO',
                  '1Y': 'DGS1',
                  '2Y': 'DGS2',
                  '3Y': 'DGS3',
                  '5Y': 'DGS5',
                  '7Y': 'DGS7',
                  '10Y': 'DGS10',
                  '20Y': 'DGS20',
                  '30Y': 'DGS30'
                  }
# Treasury Bills Constant Maturity
TREASURY_MONTHLY = {'3M': 'GS3M',
                    '10Y': 'GS10'
                    }
# Treasury Bills Secondary Market Rate
TBILL_MONTHLY = {'1M': 'TB4WK',
                 '3M': 'TB3MS',
                 '6M': 'TB6MS',
                 '1Y': 'TB1YR'
                 }
# Spread
TREASURY_10Y_MINUS_3M = 'T10Y3MM'
TREASURY_3M_MINUS_FED_RATE = 'TB3SMFFM'

# Stock Market
SP500 = 'SP500'

# Recession
# RECESSION_MONTHLY = 'USREC'  # The classical one, recession length shorter, only look at GDP
RECESSION_MONTHLY = 'USARECM'  # The newer one, recession last longer. look at other econ variables too
