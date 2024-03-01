import time

import mysql.connector
from mysql.connector import Error


def create_connection():
    connection = None
    try:
        connection = mysql.connector.connect(
            host='121.250.209.74',
            user='root',
            passwd='cong0917',
            database='metadata'
        )
        print("Connection to MySQL DB successful")
    except Error as e:
        print(f"The error '{e}' occurred")
    return connection


def execute_query(connection, query):
    cursor = connection.cursor()
    try:
        cursor.execute(query)
        connection.commit()
        print("Query executed successfully")
    except Error as e:
        print(f"The error '{e}' occurred")


def main():
    # 创建数据库连接
    connection = create_connection()

    # 创建少量示例表来演示，每个表20个字段
    # for table_number in range(1, 5000):  # 仅创建5个表作为演示
    #     create_table_query = f"""
    #     CREATE TABLE IF NOT EXISTS table_{table_number} (
    #         id INT AUTO_INCREMENT,
    #         {" ,".join(f'field_{i} INT' for i in range(1, 21))},
    #         PRIMARY KEY (id)
    #     );
    #     """
    #     execute_query(connection, create_table_query)

    # 查询创建的表的元数据
    time_start = time.time()
    query_metadata = """
    SELECT TABLE_NAME, COLUMN_NAME
    FROM INFORMATION_SCHEMA.COLUMNS
    WHERE TABLE_SCHEMA = 'metadata' AND TABLE_NAME LIKE 'table_%';
    """
    cursor = connection.cursor()
    try:
        cursor.execute(query_metadata)
        print("Tables and columns fetched successfully:")
        for (table_name, column_name) in cursor:
            print(f"Table: {table_name}, Column: {column_name}")
    except Error as e:
        print(f"The error '{e}' occurred")

    time_end = time.time()

    print('time cost', time_end - time_start, 's')

    # 关闭数据库连接
    if connection.is_connected():
        # cursor.close()
        connection.close()
        print("MySQL connection is closed")


if __name__ == "__main__":
    main()
