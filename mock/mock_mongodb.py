import random
from datetime import datetime, timedelta

import pymongo

# 连接到MongoDB数据库
client = pymongo.MongoClient("mongodb://121.250.209.74:27017/")
db = client["E-commerce"]  # 请替换成您的数据库名称
collection = db["sales_data"]  # 请替换成您的集合名称

# 商品类别列表
categories = ["Electronics", "Clothing", "Books", "Toys", "Home Appliances"]

"""
{
    "_id": ObjectId("60a7ea50745be1f6d87c8923"),  // MongoDB自动生成的文档ID
    "product_id": 12345,  // 商品ID
    "product_name": "Product Name",  // 商品名称
    "category": "Electronics",  // 商品类别
    "price": 299.99,  // 商品价格
    "quantity_sold": 100,  // 销售数量
    "order_date": ISODate("2024-01-25T14:30:00Z")  // 订单日期
}

"""

if __name__ == '__main__':
    # 生成并插入1000条模拟数据
    for i in range(1000):
        product = {
            "product_id": i + 1,
            "product_name": f"Product {i + 1}",
            "category": random.choice(categories),
            "price": round(random.uniform(10.0, 500.0), 2),  # 随机生成价格
            "quantity_sold": random.randint(1, 100),
            "order_date": datetime.now() - timedelta(days=random.randint(1, 365))
        }
        collection.insert_one(product)

    print("插入完成，共插入1000条模拟销售数据。")
