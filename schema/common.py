WARNING_ENUM = {
    '병용금기': 0,
    '특정연령대금기': 1,
    '특정연령금기': 1,
    '임부금기': 2,
    '노인주의': 3,
    '용량주의': 4,
    '투여기간주의': 5,
    '효능군중복': 6,
    '효능군중복주의': 6,
    '효능군 중복': 6,
    '효능군 중복주의': 6,
    '분할주의': 7,
    '서방정분할주의': 7,
}

MUST_COLS_ITEM = [
    'warning_type',
    'type_name_raw',
    'mix_type_raw',
    'ingr_code',
    'ingr_name_ko',
    'ingr_name_en',
    'notification_date',
    'prohbt_content_raw'
]

MUST_COLS_INGR = MUST_COLS_ITEM + [
    'related_ingr_code',
    'form_name'
]