from dataclasses import dataclass
from datetime import datetime
from logging import exception
from fauna_db import *
import dataclasses


class PirateData(FaunaDB):
    '''
    Class that interacts with the FaunaDB database, and provides helpful getter
    and setter functions that allow seemless manipulation of values for the URL 
    records stored in the PirateData table.
    '''

    ''' Structure of URL object stored in PirateData table of FaunaDB. '''
    @dataclass
    class URL:
        url_key: str
        has_pirated_content: bool
        has_udp: bool
        on_blacklist: bool
        pcd_last: float
        crawled_last: float

    def __init__(self):
        '''
        Method initializes the class by generating table indeces that will be
        used later for faster query lookup times.
        '''
        # Initialize Fauna Database
        super().__init__()
        self.table_name = 'PirateData'
        # Generate table indeces for urls for faster lookup times
        try:
            self.urls_index_name = 'urls'
            self.create_index(
                table_name=self.table_name,
                index_name=self.urls_index_name,
                key_name='url_key'
            )
        except self.FaunaDBException as exception:
            print(exception)

    def __new__(cls):
        ''' Method is used to turn the class into a Singleton. '''
        if not hasattr(cls, 'instance'):
            cls.instance = super(PirateData, cls).__new__(cls)
        return cls.instance

    ###########################################################################
    ################################ CREATE ###################################
    ###########################################################################

    def create_new_record(self, url: URL) -> dict:
        '''
        Method creates a new record in the PirateData table in the Fauna 
        database, and returns the result.
		
		:param str url: The URL key of database record
		:returns dict: The database record that was created   
        '''
        try:
            result = self.create_query_in_table(
                table_name=self.table_name,
                query_data=dataclasses.asdict(url)
            )
            return result
        except self.FaunaDBException as exception:
            raise exception

    ###########################################################################
    ################################ DELETE ###################################
    ###########################################################################

    def delete_record(self, url_key: str) -> dict:
        '''
        Method deletes the record at the given key value, and returns the 
        database row reference, else the error if there was one encountered. 
		
		:param str url_key: The URL key of database record
		:returns dict: The database record that was deleted
        '''
        try:
            result = self.delete_query_in_table_by_key(
                table_name=self.table_name,
                index_name=self.urls_index_name,
                key=url_key
            )
            return result
        except self.FaunaDBException as exception:
            raise exception

    ###########################################################################
    ############################### GETTERS ###################################
    ###########################################################################

    def get_all_url_records(self) -> dict:
        '''
        Method returns all url records from the PirateData table.

        :returns dict: url records from the PirateData table
        '''
        try:
            records = self.read_all_records_from_table(
                table_name=self.table_name
            )
            return records
        except FaunaDB.FaunaDBException as exception:
            raise exception
    
    def get_pirated_content_boolean(self, url_key: str) -> bool:
        ''' 
        Method returns the has_pirated_content value from url record. 
		
		:param str url_key: The URL key of database record
		:returns bool: The has_pirated_content value from url record
        '''
        try:
            result = self.read_query_from_table_by_key(
                index_name=self.urls_index_name,
                key=url_key
            )
            return result['data']['has_pirated_content']
        except FaunaDB.FaunaDBException as exception:
            raise exception

    def get_udp_boolean(self, url_key: str) -> bool:
        ''' 
        Method returns the has_udp value from url record. 
		
		:param str url_key: The URL key of database record
		:returns bool: The has_udp value from url record
        '''
        try:
            result = self.read_query_from_table_by_key(
                index_name=self.urls_index_name,
                key=url_key
            )
            return result['data']['has_udp']
        except FaunaDB.FaunaDBException as exception:
            raise exception

    def get_blacklist_boolean(self, url_key: str) -> bool:
        ''' 
        Method returns the on_blacklist value from url record. 
		
		:param str url_key: The URL key of database record
		:returns bool: The on_blacklist value from url record
        '''
        try:
            result = self.read_query_from_table_by_key(
                index_name=self.urls_index_name,
                key=url_key
            )
            return result['data']['on_blacklist']
        except FaunaDB.FaunaDBException as exception:
            raise exception

    def get_pcd_last(self, url_key: str) -> float:
        ''' 
        Method returns the pcd_last value from url record. 
		
		:param str url_key: The URL key of database record
		:returns float: The pcd_last timestamp value from url record
        '''
        try:
            result = self.read_query_from_table_by_key(
                index_name=self.urls_index_name,
                key=url_key
            )
            return result['data']['pcd_last']
        except FaunaDB.FaunaDBException as exception:
            raise exception

    def get_crawled_last(self, url_key: str) -> float:
        ''' 
        Method returns the crawled_last value from url record. 
		
		:param str url_key: The URL key of database record
		:returns float: The crawled_last timestamp value from url record
        '''
        try:
            result = self.read_query_from_table_by_key(
                index_name=self.urls_index_name,
                key=url_key
            )
            return result['data']['crawled_last']
        except FaunaDB.FaunaDBException as exception:
            raise exception

    ###########################################################################
    ############################### SETTERS ###################################
    ###########################################################################

    def set_new_url_key(self, old_url_key: str, new_url_key: str) -> dict:
        ''' 
        Method sets the url_key value from url record to a new value, and
        returns the result. 
		
		:param str old_url_key: The old URL key of database record
        :param str new_url_key: The new URL key of database record
		:returns dict: The database row that was updated
        '''
        try:
            result = self.update_query_in_table_by_key(
                table_name=self.table_name,
                index_name=self.urls_index_name,
                key=old_url_key,
                new_data={"url_key": new_url_key}
            )
            return result
        except FaunaDB.FaunaDBException as exception:
            raise exception

    def set_pirated_content_boolean(self, url_key: str, has_pirated_content: bool) -> dict:
        ''' 
        Method sets the has_pirated_content value from url record to a new 
        value, and returns the result. 
		
		:param str url_key: The URL key of database record
        :param bool has_pirated_content: The new has_pirated_content value of database record
		:returns dict: The database row that was updated
        '''
        try:
            result = self.update_query_in_table_by_key(
                table_name=self.table_name,
                index_name=self.urls_index_name,
                key=url_key,
                new_data={"has_pirated_content": has_pirated_content}
            )
            return result
        except FaunaDB.FaunaDBException as exception:
            raise exception

    def set_udp_boolean(self, url_key: str, has_udp: bool) -> dict:
        ''' 
        Method sets the has_udp value from url record to a new value, and 
        returns the result. 
		
		:param str url_key: The URL key of database record
        :param bool has_udp: The new has_udp value of database record
		:returns dict: The database row that was updated
        '''
        try:
            result = self.update_query_in_table_by_key(
                table_name=self.table_name,
                index_name=self.urls_index_name,
                key=url_key,
                new_data={"has_udp": has_udp}
            )
            return result
        except FaunaDB.FaunaDBException as exception:
            raise exception

    def set_blacklist_boolean(self, url_key: str, on_blacklist: bool) -> dict:
        ''' 
        Method sets the on_blacklist value from url record to a new value, and 
        returns the result. 
		
		:param str url_key: The URL key of database record
        :param bool has_udp: The new on_blacklist value of database record
		:returns dict: The database row that was updated
        '''
        try:
            result = self.update_query_in_table_by_key(
                table_name=self.table_name,
                index_name=self.urls_index_name,
                key=url_key,
                new_data={"on_blacklist": on_blacklist}
            )
            return result
        except FaunaDB.FaunaDBException as exception:
            raise exception

    def set_pcd_last(self, url_key: str, pcd_last: float) -> dict:
        ''' 
        Method sets the pcd_last value from url record to a new timestamp 
        value, and returns the result. 
		
		:param str url_key: The URL key of database record
        :param float pcd_last: The new pcd_last timestamp value of database record
		:returns dict: The database row that was updated
        '''
        try:
            result = self.update_query_in_table_by_key(
                table_name=self.table_name,
                index_name=self.urls_index_name,
                key=url_key,
                new_data={"pcd_last": pcd_last}
            )
            return result
        except FaunaDB.FaunaDBException as exception:
            raise exception

    def set_crawled_last(self, url_key: str, crawled_last: float) -> dict:
        ''' 
        Method sets the crawled_last value from url record to a new timestamp 
        value, and returns the result.
		
		:param str url_key: The URL key of database record
        :param float crawled_last: The new crawled_last timestamp value of database record
		:returns dict: The database row that was updated 
        '''
        try:
            result = self.update_query_in_table_by_key(
                table_name=self.table_name,
                index_name=self.urls_index_name,
                key=url_key,
                new_data={"crawled_last": crawled_last}
            )
            return result
        except FaunaDB.FaunaDBException as exception:
            raise exception

# Testing
'''
if __name__ == "__main__":
	# Create Instance to the PirateData in Fauna Database
	pirate_data = PirateData()

	# Instantiate Sample PirateData url
	pirate_data_url = pirate_data.URL(
		url_key = "www.yahoo.com",
		has_pirated_content = False,
		has_udp = False,
		on_blacklist = False,
		pcd_last = datetime.now().timestamp(),
		crawled_last = datetime.now().timestamp()
	)

	# Create a new record in the PirateData table in the Fauna database
	try:
		result = pirate_data.create_new_record(url=pirate_data_url)
		print(f"Created new record with key \"{result['data']['url_key']}\"!")
	except FaunaDB.FaunaDBException as exception:
		print(exception)
	
	# Set the has_pirated_content value from url record to a new value
	try:
		result = pirate_data.set_pirated_content_boolean(
			url_key=pirate_data_url.url_key,
			has_pirated_content=True
		)
		print(f"Setting value of has_pirated_content in "
			  f"\"{pirate_data_url.url_key}\" to: "
			  f"{result['data']['has_pirated_content']}")
	except FaunaDB.FaunaDBException as exception:
		print(exception)
	# Get the has_pirated_content value from url record
	try:
		result = pirate_data.get_pirated_content_boolean(
			url_key=pirate_data_url.url_key
		)
		print(f"\"{pirate_data_url.url_key}\" has pirated content: {result}")
	except FaunaDB.FaunaDBException as exception:
		print(exception)
	
	# Set the has_udp value from url record to a new value
	try:
		result = pirate_data.set_udp_boolean(
			url_key=pirate_data_url.url_key,
			has_udp=True
		)
		print(f"Setting value of has_udp in \"{pirate_data_url.url_key}\" to: "
			  f"{result['data']['has_udp']}")
	except FaunaDB.FaunaDBException as exception:
		print(exception)
	# Get the has_udp value from url record
	try:
		result = pirate_data.get_udp_boolean(
			url_key=pirate_data_url.url_key
		)
		print(f"\"{pirate_data_url.url_key}\" has UDP: {result}")
	except FaunaDB.FaunaDBException as exception:
		print(exception)

	# Set the on_blacklist value from url record to a new value
	try:
		result = pirate_data.set_blacklist_boolean(
			url_key=pirate_data_url.url_key,
			on_blacklist=True
		)
		print(f"Setting value of on_blacklist in \"{pirate_data_url.url_key}\""
			  f" to: {result['data']['on_blacklist']}")
	except FaunaDB.FaunaDBException as exception:
		print(exception)
	# Get the on_blacklist value from url record
	try:
		result = pirate_data.get_blacklist_boolean(
			url_key=pirate_data_url.url_key
		)
		print(f"\"{pirate_data_url.url_key}\" is on the blacklist: {result}")
	except FaunaDB.FaunaDBException as exception:
		print(exception)
	
	# Set the pcd_last value from url record to a new value
	try:
		new_timestamp = datetime.now().timestamp()
		result = pirate_data.set_pcd_last(
			url_key=pirate_data_url.url_key,
			pcd_last=new_timestamp
		)
		print(f"Setting value of pcd_last in \"{pirate_data_url.url_key}\" "
			  f"to a new timestamp: {result['data']['pcd_last']}")
	except FaunaDB.FaunaDBException as exception:
		print(exception)
	# Get the pcd_last value from url record
	try:
		result = pirate_data.get_pcd_last(
			url_key=pirate_data_url.url_key
		)
		timestamp = datetime.fromtimestamp(result).strftime("%c")
		print(f"\"{pirate_data_url.url_key}\" last PCD timestamp: {timestamp}")
	except FaunaDB.FaunaDBException as exception:
		print(exception)
	
	# Set the crawled_last value from url record to a new value
	try:
		new_timestamp = datetime.now().timestamp()
		result = pirate_data.set_crawled_last(
			url_key=pirate_data_url.url_key,
			crawled_last=new_timestamp
		)
		print(f"Setting value of crawled_last in \"{pirate_data_url.url_key}\""
			  f" to a new timestamp: {result['data']['crawled_last']}")
	except FaunaDB.FaunaDBException as exception:
		print(exception)
	# Get the crawled_last value from url record
	try:
		result = pirate_data.get_crawled_last(
			url_key=pirate_data_url.url_key
		)
		timestamp = datetime.fromtimestamp(result).strftime("%c")
		print(f"\"{pirate_data_url.url_key}\" last crawled timestamp: "
			  f"{timestamp}")
	except FaunaDB.FaunaDBException as exception:
		print(exception)

	# Set the url value from url record to a new value
	try:
		old_url_key = pirate_data_url.url_key
		new_url_key = "www.aol.com"
		result = pirate_data.set_new_url_key(
			old_url_key=old_url_key,
			new_url_key=new_url_key
		)
		print(f"Set the value of url_key in \"{old_url_key}\" to "
			  f"\"{new_url_key}\"")
	except FaunaDB.FaunaDBException as exception:
		print(exception)

	# Delete the record from the PirateData table in the Fauna database
	try:
		result = pirate_data.delete_record(url_key='www.google.com')
		print(f"Deleted record with key \"{result['data']['url_key']}\"!")
	except FaunaDB.FaunaDBException as exception:
		print(exception)
#'''
