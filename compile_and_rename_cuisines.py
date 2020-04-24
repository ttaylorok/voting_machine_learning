import pandas as pd

oldpath = "C:\\Users\\comp\\Documents\\OSM\\"

df1 = pd.read_csv(oldpath + "all_restaurants_usa_overpass_v1.csv")
df2 = pd.read_csv(oldpath + "all_restaurants_usa_overpass_v2.csv")
df3 = pd.read_csv(oldpath + "all_restaurants_usa_overpass_v3.csv")
df4 = pd.read_csv(oldpath + "all_restaurants_usa_overpass_v4.csv")
df5 = pd.read_csv(oldpath + "all_restaurants_usa_overpass_v5.csv")
df6 = pd.read_csv(oldpath + "all_restaurants_usa_overpass_v6.csv")
df7 = pd.read_csv(oldpath + "all_restaurants_usa_overpass_v7.csv")
df8 = pd.read_csv(oldpath + "all_restaurants_usa_overpass_v8.csv")
df9 = pd.read_csv(oldpath + "all_restaurants_usa_overpass_v9.csv")
data_ff = pd.read_csv(oldpath + "all_restaurants_usa_overpass_fast_food_v1.csv")
data_ff['fastfood'] = 'yes'

dfc = df1.append(df2).append(df3).append(df4).append(df5).append(df6).append(df7).append(df8).append(df9)
dfc['fastfood'] = 'no'

dff = dfc.append(data_ff)

dff.drop_duplicates(inplace=True)

dff.columns = ["id","name","cuisine","lat","lon","state","fastfood"]




filt = dff["cuisine"] != "none"
dff = dff.loc[filt,:]

dff["cuisine_new"] = dff["cuisine"]
dff['row_id'] = 0
dff["cuisine_group"] = "none"
row_id = 0
"""
classifications to add: 
  cont = asia, europe, africa, middle east, latin america
  ethnic = yes/no (foreign food or no)
  https://en.wikipedia.org/wiki/List_of_cuisines
  group = according to wiki
"""

for index, row in dff.iterrows():
    if row_id % 1000 == 0 : print(row_id)
    c = row["cuisine"].lower()
    group = "none"
    if "mexican" in c or "taco" in c or "burrito" in c or "taqueria" in c or "yucatan" in c or "taquiera" in c:
        new = "mexican"
        group = "America, Northern"
        
    elif "vietnamese" in c or "vitamise" in c:
        new = "vietnamese"
        group = "Asia, Southeastern"
        
    elif "tex-mex" in c:
        new = "tex-mex"
        group = "America, Northern"
        
    elif "italian" in c:
        new = "italian"
        group = "European, Southern"
    
    elif "mediterranean" in c or "mediteran" in c:
        new = "mediterranean"
        group = "European, Southern"
        
    elif "japanese" in c or "hibachi" in c or "habachi" in c or "okinawan" in c:
        new = "japanese"
        group = "Asia, Eastern"
        
    elif "hungarian" in c:
        new = "hungarian"
        group = "European, Central"
        
    elif "burmese" in c:
        new = "burmese"
        group = "Asia, Southeastern"
    
    elif "ecuadorian" in c or "ecuatorian" in c or "ecuadorean" in c:
        new = "ecuadorian"
        group = "America, Southern"
        
    elif "indian" in c:
        new = "indian"
        group = "Asia, Southern"
    
    elif "french" in c:
        new = "french"
        group = "European, Western"
        
    elif "thai" in c:
        new = "thai"
        group = "Asia, Southeastern"
        
    elif "korean" in c:
        new = "korean"
        group = "Asia, Eastern"

    elif "nepalese" in c or "nepal" in c or "himalayan" in c:
        new = "nepalese"
        group = "Asia, Southern"
        
    elif "sushi" in c:
        new = "sushi"
        group = "Asia, Eastern"
        
    elif "chinese" in c or "china" in c:
        new = "chinese"  
        group = "Asia, Eastern"
        
    elif "cajun" in c:
        new = "cajun"
        group = "America, Northern"
        
    elif "taiwan" in c:
        new = "taiwanese"
        group = "Asia, Eastern"
        
    elif "wings" in c or "chicken" in c:
        new = "chicken"
        group = "America, Northern"
        
    elif "greek" in c:
        new = "greek"
        group = "European, Southern"
        
    elif "hawaiian" in c or "hawaiin" in c or "hawiian" in c or "poke" in c:
        new = "hawaiian"
        group = "America, Northern"
        
    elif "persian" in c:
        new = "persian"
        group = "Asia, Western"
        
    elif "turkish" in c:
        new = "turkish" 
        group = "Asia, Western"
        
    elif "peruvian" in c or "peru" in c:
        new = "peruvian"
        group = "America, Southern"
        
    elif "german" in c or "bavarian" in c:
        new = "german"
        group = "European, Central"
        
    elif "catfish" in c:
        new = "catfish"
        group = "America, Northern"
        
    elif "brazilian" in c or "brazillian_grill" in c or "brazillian" in c:
        new = "brazilian"
        group = "America, Southern"
        
    elif "ramen" in c:
        new = "ramen"
        group = "none"
        
    elif "vegan" in c:
        new = "vegan"
        group = "none"
        
    elif "argentinian" in c or "argentine" in c:
        new = "argentinian"
        group = "America, Southern"
        
    elif "ethiopian" in c or "ethopia" in c:
        new = "ethiopian"
        group = "Africa, Eastern"
    
    elif "irish" in c:
        new = "irish"
        group = "European, Northern"
        
    elif "pasta" in c:
        new = "pasta"
        group = "none"
        
    elif "tapas" in c:
        new = "tapas"
        group = "European, Southern"
        
    elif "spanish" in c or "spainish" in c:
        new = "spanish"
        group = "European, Southern"
        
    elif "mongolian" in c:
        new = "mongolian"
        group = "Asia, Eastern"
        
    elif "russian" in c:
        new = "russian"
        group = "Asia, Central"
        
    elif "hot_dog" in c or "hotdogs" in c:
        new = "hot_dog"
        group = "America, Northern"
        
    elif "fish_and_chips" in c:
        new = "fish_and_chips"
        group = "none"
        
    elif "lebanese" in c:
        new = "lebanese"
        group = "Asia, Western"
        
    elif "costa_rican" in c:
        new = "costa_rican"
        group = "America, Central"
        
    elif "african" in c:
        new = "african"
        group = "none"
        
    elif "malaysian" in c:
        new = "malaysian"
        group = "Asia, Southeastern"
        
    elif "guyanese" in c:
        new = "guyanese"
        group = "Caribbean"
        
    elif "vegetarian" in c or "vegeterian" in c:
        new = "vegetarian"
        group = "none"
    
    elif "afghan" in c or "afgan" in c:
        new = "afghan"
        group = "Asia, Southern"
        
    elif "salvadorean" in c or "salvadoran" in c or "salvadorian" in c or "salvador" in c:
        new = "el salvadorean"
        group = "America, Central"
        
    elif "teriyaki" in c or "teryiaki" in c:
        new = "teriyaki"
        group = "Asia, Eastern"
        
    elif "moroccan" in c:
        new = "moroccan"
        group = "Africa, Northern"
    
    elif "Caribbean" in c:
        new = "Caribbean"
        group = "Caribbean"
        
    elif "pho" in c:
        new = "pho"
        group = "Asia, Southeastern"
        
    elif "noodle" in c:
        new = "noodle"
        group = "none"
        
    elif "belgian" in c:
        new = "belgian"
        group = "European, Western"
        
    elif "jamaican" in c or "jamiacan" in c or "jamacian" in c or "jamaica" in c:
        new = "jamaican"
        group = "Caribbean"
        
    elif "kebab" in c:
        new = "kebab"
        group = "Asia, Western"
        
    elif "african" in c:
        new = "african"
        group = "african"
        
    elif "fondue" in c:
        new = "fondue"
        group = "none"
    
    elif "swedish" in c:
        new = "swedish"
        group = "European, Northern"
        
    elif "cuban" in c:
        new = "cuban"
        group = "Caribbean"
        
    elif "nicaraguan" in c:
        new = "nicaraguan"
        group = "America, Central"
    
    elif "philippine" in c or "filipino" in c or "philipino" in c:
        new = "philippine"
        group = "Asia, Southeastern"
        
    elif "pub" in c:
        new = "pub"
        group = "none"
        
    elif "guamian" in c:
        new = "guamian"
        group = "Oceanic"
        
    elif "lithuanian" in c:
        new = "lithuanian"
        group = "European, Northern"
        
    elif "deli" in c:
        new = "deli"
        group = "none"
    
    elif "lao" in c or "loatian" in c:
        new = "lao"
        group ="Asia, Southeastern"
        
    elif "indonesian" in c:
        new = "indonesian"
        group = "Asia, Southeastern"
        
    elif "kenyan" in c:
        new = "kenyan"
        group = "Africa, Eastern"
    
    elif "ghanaian" in c:
        new = "ghanaian"
        group = "African, Western"
        
    elif "australian" in c:
        new = "australian"
        
    elif "egyptian" in c or "egytian" in c:
        new = "egyptian"
        group = "Africa, Northern"
        
    elif "yemen" in c:
        new = "yemen"
        group = "Asia, Western"
        
    elif "english" in c or "british" in c:
        new = "english"
        group = "European, Northern"
        
    elif "tibetan" in c:
        new = "tibetan"
        group = "Asia, Eastern"
        
    elif "creole" in c:
        new = "creole"
        group = "America, Northern"
        
    elif "uzbek" in c:
        new = "uzbek"
        group = "Asia, Central"
        
    elif "crepe" in c:
        new = "crepe"
        group = "none"
        
    elif "colombian" in c or "columbian" in c:
        new = "colombian"
        group = "America, Southern"
        
    elif "pakistani" in c:
        new = "pakistani"
        group = "Asia, Southern"
        
    elif "polish" in c:
        new = "polish"
        group = "European, Central"
        
    elif "portuguese" in c or "portugese" in c:
        new = "portuguese"
        group = "European, Southern"
    
    elif "gyros" in c:
        new = "gyros"
        group = "European, Southern"
    
    elif "somali" in c:
        new = "somali"
        group = "Africa, Eastern"

    elif "dutch" in c:
        new = "dutch"
        group = "European, Western"

    elif "panamanian" in c:
        new = "panamanian"
        group = "America, Central"
        
    elif "dominican" in c:
        new = "dominican"
        group = "Caribbean"
        
    elif "uyghur" in c:
        new = "uyghur"
        group = "Asia, Eastern"
        
    elif "health" in c:
        new = "health"
        group = "none"
        
    elif "bolivian" in c:
        new = "bolivian"
        group = "America, Southern"
        
    elif "bosnian" in c:
        new = "bosnian"
        group = "European, Southern"
        
    elif "latin" in c or "hispanic" in c:
        new = "latin"
        group = "none"
        
    elif "scandinavian" in c:
        new = "scandinavian"
        group = "European, Northern"
        
    elif "venezuelan" in c or "venezuelen" in c or "venuzuelan" in c:
        new = "venezuelan"
        group = "America, Southern"
        
    elif "health" in c:
        new = "health"
        group = "none"
        
    elif "senegalese" in c:
        new = "senegalese"
        group = "African, Western"

    elif "cambodian" in c:
        new = "cambodian"
        group = "Asia, Southeastern"
        
    elif "singaporean" in c:
        new = "singaporean"
        group = "Asia, Southeastern"
    
    elif "iraqi" in c:
        new = "iraqi"
        group = "Asia, Western"
        
    elif "new_zealand" in c:
        new = "new_zealand"
        group = "none"
        
    elif "swiss" in c:
        new = "swiss"
        group = "European, Western"
        
    elif "polynesian" in c:
        new = "polynesian"
        group = "Oceanic"
        
    elif "croatian" in c:
        new = "croatian"
        group = "European, Southern"
        
    elif "guatemalan" in c or "guatamalen" in c or "guatemalteca" in c:
        new = "guatemalan"
        group = "America, Central"
        

    elif "romanian" in c:
        new = "romanian"
        group = "European, Southern"
        
    elif "serbian" in c:
        new = "serbian"
        group = "European, Southern"
        
    elif "czech" in c:
        new = "czech"
        group = "European, Central"
        
    elif "sri_lankan" in c:
        new = "sri_lankan"
        group = "Asia, Southern"
        
    elif "chilean" in c:
        new = "chilean"
        group = "America, Southern"
        
    elif "szechuan" in c:
        new = "szechuan"
        group = "Asia, Eastern"
        
    elif "norwegian" in c:
        new = "norwegian"
        group = "European, Northern"
        
    elif "haitian" in c:
        new = "haitian"
        group = "Caribbean"
        
    elif "shanghainese" in c:
        new = "shanghainese"
        group = "Asia, Eastern"
        
    elif "danish" in c:
        new = "danish"
        group = "European, Northern"
        
    elif "ukrainian" in c:
        new = "ukrainian"
        group = "European, Eastern"
        
    elif "georgian" in c:
        new = "georgian"
        group = "European, Eastern"
        
    elif "uruguayan" in c:
        new = "uruguayan"
        group = "America, Southern"
        
    elif "cambodia" in c:
        new = "cambodia"
        group = "Asia, Southeastern"
        
    elif "hong_kong" in c or "hong kong" in c:
        new = "hong_kong"
        group = "Asia, Eastern"
        
    elif "balkan" in c:
        new = "balkan"
        group = "European, Southern"
        
    elif "cantonese" in c:
        new = "cantonese"
        group = "Asia, Eastern"
        
    elif "nordic" in c:
        new = "nordic"
        group = "European, Northern"
        
    elif "nigerian" in c:
        new = "nigerian"
        group = "African, Western"
        
    elif "honduran" in c:
        new = "honduran"
        group = "America, Central"
        
    elif "puerto_rican" in c:
        new = "puerto_rican"
        group = "America, Northern"
        
    elif "syrian" in c:
        new = "syrian"
        group = "Asia, Western"
        
    elif "belizean" in c:
        new = "belizean"
        group = "America, Central"
        
    elif "israeli" in c or "jewish" in c or "kosher" in c:
        new = "israeli"
        group = "Asia, Western"

    elif "austrian" in c:
        new = "austrian"
        group = "European, Central"
        
    elif "armenian" in c:
        new = "armenian"
        group = "European, Eastern"
        
    elif "bakery" in c or "cookies" in c or "cake" in c or "cupcake" in c or "pastry" in c or "bread" in c:
        new = "bakery"
    elif "cafe" in c:
        new = "cafe"
    elif "pizza" in c:
        new = "pizza"
    elif "donut" in c or "doughnut" in c:
        new = "donut"
    elif "juice" in c or "smoothie" in c:
        new = "juice"
    elif "dessert" in c:
        new = "dessert"
    elif "buffet" in c:
        new = "buffet"
    elif "pie" in c:
        new = "pie"
    elif "sandwich" in c or "sub" in c or "sandwhich" in c:
        new = "sandwich"
    elif "frozen_yogurt" in c:
        new = "frozen_yogurt"
    elif "macaroni_and_cheese" in c:
        new = "macaroni_and_cheese"
    elif "falafel" in c:
        new = "falafel"
    elif "chili" in c:
        new = "chili"
    elif "crab" in c:
        new = "crab"
    elif "burger" in c:
        new = "burger"
    elif "bbq" in c or "barbeque" in c or "barbecue" in c or "bar-b-q" in c or "smokehouse" in c:
        new = "bbq"
    elif "breakfast" in c or "pancake" in c or "waffle" in c or "omelet" in c:
        new = "breakfast"
    elif "home" in c:
        new = "home"
    elif "coffee" in c:
        new = "coffee"
    elif "ice_cream" in c:
        new = "ice_cream"
    elif "fast_food" in c or "fast food" in c or "fastfood" in c:
        new = "fast_food"
    elif "bagel" in c:
        new = "bagel"
    elif "southern" in c:
        new = "southern"
    elif "steak" in c:
        new = "steak"
    elif "soul" in c:
        new = "soul"
    elif "soup" in c:
        new = "soup"
    elif "salad" in c:
        new = "salad"
    elif "sausage" in c:
        new = "sausage"
    elif "home" in c or "comfort" in c:
        new = "home"
    elif "diner" in c:
        new = "diner"
    elif "western" in c:
        new = "western"
    elif "northwest" in c or "north-western" in c:
        new = "northwest" 
    elif "southwest" in c:
        new = "southwest"
    elif "oysters" in c:
        new = "oysters"
    elif "fish" in c:
        new = "fish"
    elif "lobster" in c:
        new = "lobster"
    elif "regional" in c:
        new = "regional"
    elif "coney_island" in c or "coney island" in c or "coney" in c:
        new = "coney_island"
    elif "grilled_cheese" in c:
        new = "grilled_cheese"
    elif "fine_dining" in c or "fine dining" in c:
        new = "fine_dining"
    elif "bar&grill" in c or "Bar_&_Ggrill" in c or "Bar_&_Ggrill" in c or "grill" in c:
        new = "bar_and_grill"
    elif "seafood" in c:
        new = "seafood"
    elif "bar" in c or "beer" in c or "bourbon" in c or "wine" in c or "tavern" in c or "speakeasy" in c:
        new = "bar"
    elif "middle_eastern" in c or "arab" in c or "middle east" in c:
        new = "middle_eastern"
    elif "international" in c or "world" in c:
        new = "international"
    elif "european" in c:
        new = "european"
    elif "america" in c:
        new = "american"
    elif "asian" in c:
        new = "asian"
    else:
        new = "other"
    dff.iloc[row_id,7] = new
    dff.iloc[row_id,8] = row_id
    dff.iloc[row_id,9] = group
    row_id += 1


#dff.to_csv("all_restaurants_compiled_full.csv")
dff.to_csv("all_restaurants_and_ff_compiled_with_cuisine.csv")

# filt2 = dff["cuisine_new"] == "other"
# dff.loc[filt2,"cuisine"]

# g = dff.groupby(["state","cuisine_new"])["cuisine_new"].count()
# g2 = g.groupby("state")["cuisine_new"].count()

# bins = dff.groupby("cuisine_new")["cuisine"].unique()
# bins.to_csv("bins.csv")
