#3.1. Wrangling Data with MongoDB

from pprint import PrettyPrinter

import pandas as pd
from IPython.display import VimeoVideo
from pymongo import MongoClient

#Task 3.1.1: Instantiate a PrettyPrinter, and assign it to the variable pp.
pp = PrettyPrinter(indent=2)

#PREPARE DATA
#CONNECT

#Task 3.1.2: Create a client that connects to the database running at localhost on port 27017.
client = MongoClient(host="localhost", port=27017)

#EXPLORE
#Task 3.1.3: Print a list of the databases available on client.
client.list_databases() # returns an iterator

from sys import getsizeof
my_list = [0,1,2,3,4] # lists take more space than ranges
my_range= range(0,5) #iterator
for i in my_range:
    print(i)

pp.pprint(list(client.list_databases())) # the db that we need is 'air-quality'

#Task 3.1.4: Assign the "air-quality" database to the variable db.
db = client["air-quality"]

#Task 3.1.5: Use the list_collections method to print a list of the collections available in db.
for c in db.list_collections():
    print(c["name"])
# we need the nairobi collection

#Task 3.1.6: Assign the "nairobi" collection in db to the variable name nairobi.
nairobi = db["nairobi"]

#Task 3.1.7: Use the count_documents method to see how many documents are in the nairobi collection.
nairobi.count_documents({}) #202212

#Task 3.1.8: Use the find_one method to retrieve one document from the nairobi collection, and assign it to the variable name result.
result = nairobi.find_one({})
pp.pprint(result)

#Task 3.1.9: Use the distinct method to determine how many sensor sites are included in the nairobi collection.
nairobi.distinct("metadata.site")

#Task 3.1.10: Use the count_documents method to determine how many readings there are for each site in the nairobi collection.
print("Documents from site 6:", nairobi.count_documents({"metadata.site":6})) #Documents from site 6: 70360
print("Documents from site 29:",nairobi.count_documents({"metadata.site":29})) #Documents from site 29: 131852

#Task 3.1.11: Use the aggregate method to determine how many readings there are for each site in the nairobi collection.
result = nairobi.aggregate(
    [
        {"$group": {"_id": "$metadata.site", "count": {"$count": {}}}}
    ])
pp.pprint(list(result)) # [{'_id': 6, 'count': 70360}, {'_id': 29, 'count': 131852}]

#Task 3.1.12: Use the distinct method to determine how many types of measurements have been taken in the nairobi collection.
nairobi.distinct("metadata.measurement") #['P1', 'temperature', 'humidity', 'P2']
#Task 3.1.13: Use the find method to retrieve the PM 2.5 readings from all sites. Be sure to limit your results to 3 records only.
result = nairobi.find({"metadata.measurement": "P2"}).limit(3)
pp.pprint(list(result))

#Task 3.1.14: Use the aggregate method to calculate how many readings there are for each type ("humidity", "temperature", "P2", and "P1") in site 6.
result = nairobi.aggregate(
    [
        {"$match": {"metadata.site": 6}},
        {"$group": {"_id": "$metadata.measurement", "count": {"$count": {}}}}
    ])
pp.pprint(list(result)) 

#Task 3.1.15: Use the aggregate method to calculate how many readings there are for each type ("humidity", "temperature", "P2", and "P1") in site 29.
result = nairobi.aggregate(
    [
        {"$match": {"metadata.site": 29}},
        {"$group": {"_id": "$metadata.measurement", "count": {"$count": {}}}}
    ])
pp.pprint(list(result)) 

#IMPORT

#Task 3.1.16: Use the find method to retrieve the PM 2.5 readings from site 29. Be sure to limit your results to 3 records only. Since we won't need the metadata for our model, use the projection argument to limit the results to the "P2" and "timestamp" keys only.
result = nairobi.find(
    {"metadata.site": 29, "metadata.measurement": "P2"},
    projection={"P2":1, "timestamp":1, "_id":0}
)
pp.pprint(result.next()) #{'P2': 34.43, 'timestamp': datetime.datetime(2018, 9, 1, 0, 0, 2, 472000)}

#Task 3.1.17: Read records from your result into the DataFrame df. Be sure to set the index to "timestamp".
df = pd.DataFrame(result).set_index("timestamp")
df.head()