import oracledb

try:
    conn = oracledb.connect(
        user="kim1",
        password="1",
        dsn="192.168.0.80:1521/XE"
    )

    cursor = conn.cursor()
    cursor.execute("SELECT ITEM_SEQ, ITEM_NAME FROM REF_DRUG_PERMIT_LIST")

    for row in cursor:
        print(row)

    cursor.close()
    conn.close()
    print("Oracle 연결 성공")

except Exception as e:
    print("Oracle 연결 실패")
    print(e)