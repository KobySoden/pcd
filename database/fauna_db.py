import faunadb
import os
from faunadb import query as q
from faunadb.client import FaunaClient
from dotenv import load_dotenv

class FaunaDB(object):
    ''' 
    Singleton class used to interface with Fauna Database. Features full set of 
    CRUD features that interact with the database.
    '''

    class FaunaDBException(Exception):
        ''' Raised when an exception occurs with the FaunaDB operations. '''
        pass

    def __init__(self):
        ''' 
        Method initializes the class by first storing a reference of the 
        database into the global client variable.

        NOTE: Use the correct domain for your database's Region Group from here
        https://docs.fauna.com/fauna/current/learn/understanding/region_groups
        '''
        load_dotenv()
        self.client = FaunaClient(
            secret=os.getenv("FAUNA_SECRET_KEY"),
            domain="db.us.fauna.com",
            port=443,
            scheme="https"
        )

    def __new__(cls):
        ''' Method is used to turn the class into a Singleton. '''
        if not hasattr(cls, 'instance'):
            cls.instance = super(FaunaDB, cls).__new__(cls)
        return cls.instance

    ###########################################################################
    ################################ CREATE ###################################
    ###########################################################################

    def create_index(self, table_name: str, index_name: str, key_name: str) -> dict:
        ''' 
        Creates a new table index for specified table. Indeces act as a lookup 
        table that improves the performance of finding queries. 

        :param str table_name: Name of database table to create index
        :param str index_name: Name of new index to be created
        :param str key_name: Name of index key to be looked up
        :returns dict: The newly created database index
        '''
        try:
            result = self.client.query(
                q.create_index({
                    "name": index_name,
                    "source": q.collection(table_name),
                    "terms": [{'field': ['data', key_name]}],
                    "unique": True  # URLs should be unique
                })
            )
            return result
        except faunadb.errors.BadRequest as exception:
            error = exception.errors[0]
            raise self.FaunaDBException(
                f"Error 409: \"{index_name}\" {error.description}"
            )

    def create_query_in_table(self, table_name: str, query_data: dict) -> dict:
        ''' 
        Creates a new database row with the specified data within the specified
        table and returns the new database row reference, else the error if 
        there was one encountered. 

        :param str table_name: Name of database table for query to be created
        :param dict query_data: Data contained in the new query
        :returns dict: The newly created database row 
        '''
        try:
            return self.client.query(
                q.create(q.collection(table_name), {"data": query_data})
            )
        except faunadb.errors.BadRequest as exception:
            error = exception.errors[0]
            raise self.FaunaDBException(
                f"Error 409: \"{query_data['url_key']}\" {error.description}"
            )

    ###########################################################################
    ################################# READ ####################################
    ###########################################################################

    def read_query_from_table_by_key(self, index_name: str, key: str) -> dict:
        ''' 
        Reads the database row at the given key value within the specified 
        table and returns the database row reference, else the error if there 
        was one encountered. 

        :param str index_name: Name of index to be looked up
        :param str key: Key from Index to be looked up
        :returns dict: The database row that was read   
        '''
        try:
            return self.client.query(q.get(q.match(q.index(index_name), key)))
        except (faunadb.errors.BadRequest, faunadb.errors.NotFound) as exception:
            error = exception.errors[0]
            raise self.FaunaDBException(
                f"Error 409: \"{key}\" {error.description}"
            )

    def read_query_from_table_by_ref(self, table_name: str, reference_value: str) -> dict:
        ''' 
        Reads the database row at the given reference value within the 
        specified table and returns the database row reference, else the error 
        if there was one encountered.

        :param str table_name: Name of database table for query to be read
        :param str reference_value: Reference value of query
        :returns dict: The database row that was read  
        '''
        try:
            result = self.client.query(
                q.get(q.ref(q.collection(table_name), reference_value))
            )
            return result
        except (faunadb.errors.BadRequest, faunadb.errors.NotFound) as exception:
            error = exception.errors[0]
            raise self.FaunaDBException(
                f"Error 409: \"{reference_value}\" {error.description}"
            )

    def read_all_records_from_table(self, table_name: str) -> dict:
        '''
        Method returns all records from a specified table.

        :param str table_name: Name of database table that will be scraped
        :returns [dict]: Dictionary of records
        '''
        try:
            # Grab all references to the records at the specified table
            results = self.client.query(
                q.paginate(q.documents(q.collection(table_name)))
            )
            # Grab data from the table with the reference values
            records = {}
            for result in results['data']:
                record = self.read_query_from_table_by_ref(
                    table_name=table_name, reference_value=result.id()
                )
                records[record['data']['url_key']] = record
            return records
        except FaunaDB.FaunaDBException as exception:
            raise exception

    ###########################################################################
    ################################ UPDATE ###################################
    ###########################################################################

    def update_query_in_table_by_key(self, table_name: str, index_name: str, key: str, new_data: dict) -> dict:
        '''
        Updates the database row at the given key value within the 
        specified table and returns the database row reference, else the error 
        if there was one encountered. If the dictionary key/value in new_data 
        does not exist, it will create a new one.

        :param str table_name: Name of database table for query to be updated
        :param str index_name: Name of index to be looked up
        :param str key: Key from Index to be looked up  
        :param dict new_data: New data to be updated in the query
        :returns dict: The newly updated database row 
        '''
        try:
            reference_value = self.get_ref_by_key(
                index_name=index_name, key=key)
            return self.update_query_in_table_by_ref(
                table_name=table_name,
                reference_value=reference_value,
                new_data=new_data
            )
        except FaunaDB.FaunaDBException as exception:
            raise exception

    def update_query_in_table_by_ref(self, table_name: str, reference_value: str, new_data: dict) -> dict:
        '''
        Updates the database row at the given reference value within the 
        specified table and returns the database row reference, else the error 
        if there was one encountered. If the dictionary key/value in new_data 
        does not exist, it will create a new one.

        :param str table_name: Name of database table for query to be updated
        :param str reference_value: Reference value of query    
        :param dict new_data: New data to be updated in the query
        :returns dict: The newly updated database row 
        '''
        try:
            result = self.client.query(
                q.update(
                    q.ref(q.collection(table_name), reference_value),
                    {"data": new_data}
                )
            )
            return result
        except (faunadb.errors.BadRequest, faunadb.errors.NotFound) as exception:
            error = exception.errors[0]
            raise self.FaunaDBException(
                f"Error 409: \"{reference_value}\" {error.description}"
            )

    ###########################################################################
    ################################ DELETE ###################################
    ###########################################################################

    def delete_query_in_table_by_key(self, table_name: str, index_name: str, key: str) -> dict:
        ''' 
        Deletes the database row at the given key value within the specified 
        table and returns the database row reference, else the error if there 
        was one encountered. 

        :param str table_name: Name of database table for query to be deleted
        :param str index_name: Name of index to be looked up
        :param str key: Key from Index to be looked up 
        :returns dict: The newly deleted database row        
        '''
        try:
            reference_value = self.get_ref_by_key(
                index_name=index_name, key=key)
            return self.delete_query_in_table_by_ref(
                table_name=table_name,
                reference_value=reference_value
            )
        except FaunaDB.FaunaDBException as exception:
            raise exception

    def delete_query_in_table_by_ref(self, table_name: str, reference_value: str) -> dict:
        ''' 
        Deletes the database row at the given reference value within the 
        specified table and returns the database row reference, else the error 
        if there was one encountered. 

        :param str table_name: Name of database table for query to be deleted
        :param str reference_value: Reference value of query 
        :returns dict: The newly deleted database row       
        '''
        try:
            result = self.client.query(
                q.delete(q.ref(q.collection(table_name), reference_value))
            )
            return result
        except (faunadb.errors.BadRequest, faunadb.errors.NotFound) as exception:
            error = exception.errors[0]
            raise self.FaunaDBException(
                f"Error 409: \"{reference_value}\" {error.description}"
            )

    ###########################################################################
    ################################# MISC ####################################
    ###########################################################################

    def get_ref_by_key(self, index_name: str, key: str) -> str:
        ''' 
        Returns the database row reference value at the given key value, else 
        the error if there was one encountered. 

        :param str index_name: Name of index to be looked up
        :param str key: Key from Index to be looked up
        :returns str: The database row reference value
        '''
        try:
            result = self.read_query_from_table_by_key(
                index_name=index_name, key=key
            )
            return result['ref'].id()
        except FaunaDB.FaunaDBException as exception:
            raise exception

# Testing

#if __name__ == "__main__":
"""    # Database
    db = FaunaDB()

    # CREATE
    try:
        result = db.create_query_in_table(
            table_name='PirateData',
            query_data={
                "url_key":"www.google.com/",
                "last_scanned_date": "MM-DD-YYYY", 
                "has_pirated_content": False
            }
        )
        print(f"Created new database entry: {result}")
    except FaunaDB.FaunaDBException as exception:
        print(exception)

    # READ
    try:
        result = db.read_query_from_table_by_ref(
            table_name='PirateData',
            reference_value="326817979669938249"
        )
        print(result)
    except FaunaDB.FaunaDBException as exception:
        print(exception)
    try:
        result = db.read_query_from_table_by_key(
            index_name='urls',
            key='www.google.coms'
        )
        print(result)
    except FaunaDB.FaunaDBException as exception:
        print(exception)

    # UPDATE
    try:
        result = db.update_query_in_table_by_ref(
            table_name='PirateData',
            reference_value="326817979669938249",
            new_data={"detected_pirated_content": False}
        )
        print(result)
    except FaunaDB.FaunaDBException as exception:
        print(exception)
    try:
        result = db.update_query_in_table_by_key(
            table_name='PirateData',
            index_name='urls',
            key='www.google.com',
            new_data={"detected_pirated_content": True}
        )
        print(result)
    except FaunaDB.FaunaDBException as exception:
        print(exception)

    # DELETE
    try:
        result = db.delete_query_in_table_by_ref(
            table_name='PirateData',
            reference_value="326821757185949769"
        )
        print(result)
    except FaunaDB.FaunaDBException as exception:
        print(exception)
    try:   
        result = db.delete_query_in_table_by_key(
            table_name='PirateData',
            index_name='urls',
            key='www.google.com'
        )
        print(result)
    except FaunaDB.FaunaDBException as exception:
        print(exception)

    # MISC
    try:
        result = db.get_ref_by_key(index_name='urls', key='www.google.com/')
        print(result)
    except FaunaDB.FaunaDBException as exception:
        print(exception)

"""