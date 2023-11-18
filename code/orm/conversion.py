import re

import measurement.measures.volume
from measurement.utils import guess
from sqlalchemy import Column, Integer, Numeric, String

from config.logger import logger
from models.database import Base, db_session, default_language
from models.reference import ReferenceIndex

# params
gram_pattern = re.compile(r"( g)$|(?<=\d)g|( g )")


def basic_conversion(unit, number):
    """Conversion.get_weight("test", "KG", 1.2) :
    convert standards (imperial, metrics...) measure
    for weight and volume in gramme and liter"""
    try:
        m = guess(number, unit)
        if type(m) is measurement.measures.Volume:
            number = m.l
            unit = "l"
        elif type(m) is measurement.measures.Weight:
            number = m.g
            unit = "g"
    except ValueError:
        unit = unit.lower()

    return {"unit": unit, "number": number}


class Conversion(Base):
    __tablename__ = "ingredients_conversion"
    id = Column(Integer, primary_key=True)
    ingredient_name = Column(String(255))
    unit = Column(String(255))
    weight = Column(Numeric(10, 3))
    language = Column(String(4))

    def __init__(self, name, unit, weight=0, language=default_language):
        self.ingredient_name = name
        self.unit = unit
        self.weight = weight
        self.language = language

    @classmethod
    def get_weight(cls, name, unit, number, language=default_language):
        clean_measure = basic_conversion(unit, number)
        key = (
            db_session.query(cls.weight)
            .filter_by(ingredient_name=name)
            .filter_by(unit=clean_measure["unit"].lower().strip())
            .filter_by(language=language)
            .first()
        )
        if key is not None:
            return float(key[0]) * float(clean_measure["number"])
        return number

    @classmethod
    def get_ingredients_names(cls):
        return (
            db_session.query(cls.ingredient_name)
            .distinct()
            .order_by(cls.ingredient_name)
            .all()
        )


# function to convert kitchen / customal unit in dict_list into L and g
def from_kitchen_to_physics(dict_list):
    tablespoon_tuples = ReferenceIndex.get_reference_list("tablespoon_unit")
    tablespoon_unit = [item for t in tablespoon_tuples for item in t]
    teaspoon_tuples = ReferenceIndex.get_reference_list("teaspoon_unit")
    teaspoon_unit = [item for t in teaspoon_tuples for item in t]
    glass_tuples = ReferenceIndex.get_reference_list("glass_unit")
    glass_unit = [item for t in glass_tuples for item in t]
    kg_tuples = ReferenceIndex.get_reference_list("kg_unit")
    kg_unit = [item for t in kg_tuples for item in t]
    spices_tuples = ReferenceIndex.get_reference_list("spices")
    spices_list = [item for t in spices_tuples for item in t]
    new_dict_list = []
    # search for string match from unit_list in dict :
    # and format output of unit in order to spare quantity
    for ingredient_dict in dict_list:
        new_dict = {}
        for key, value in ingredient_dict.items():
            new_value = ""
            quantity_str = re.findall(r"([\d.,]+)\s*", value)
            unit = re.findall(r"([^\d\/.,]+)\s*", value)
            key = re.sub(r"([\d\/.,]+)\s*", "", key)
            if quantity_str and quantity_str != ".":
                quantity_str = quantity_str[0]
                quantity_str = re.sub(r"(1/2)", "0.5", quantity_str)
                quantity_str = re.sub(r"(½)", "0.5", quantity_str)
                quantity_str = re.sub(r"(1/3)", "0.33", quantity_str)
                quantity_str = re.sub(r"(1/4)", "0.25", quantity_str)
                quantity_str = re.sub(r"(1/8)", "0.125", quantity_str)
                quantity_str = re.sub(r"(,)", ".", quantity_str)
                try:
                    quantity_num = float(quantity_str)
                except Exception as e:
                    logger.warning(f"Error : {e}")
                    continue
            else:
                quantity_num = 1
            if unit:
                if re.search(r"(?i)(" + "|".join(glass_unit) + r")", value):
                    quantity_num = quantity_num * 0.2
                    unit_replace = " L"
                elif re.search(
                    r"(?i)(" + "|".join(teaspoon_unit) + r")", value
                ):
                    quantity_num = quantity_num * 0.005
                    unit_replace = " L"
                elif re.search(gram_pattern, value):
                    unit_replace = " g"
                elif re.search(
                    r"(?i)(" + "|".join(tablespoon_unit) + r")", value
                ):
                    quantity_num = quantity_num * 0.015
                    unit_replace = " L"
                elif re.search(r"(?i)(" + "|".join(kg_unit) + r")", value):
                    quantity_num = quantity_num * 1000
                    unit_replace = " g"
                elif re.search(r"(cl )(?![A-z])", value):
                    quantity_num = quantity_num * 0.01
                    unit_replace = " L"
                elif re.search(r"(ml)", value):
                    quantity_num = quantity_num * 0.001
                    unit_replace = " L"
                elif re.search(r"(gousse)", value):
                    quantity_num = quantity_num * 7
                    unit_replace = " g"
                elif re.search(r"(tete)", value):
                    quantity_num = quantity_num * 80
                    unit_replace = " g"
                elif re.search(r"(oeuf)", value):
                    quantity_num = quantity_num * 60
                    unit_replace = " g"
                elif re.search(r"([L][^cl])", value):
                    unit_replace = " L"
                else:
                    unit_replace = " piece"
                if re.search(r"(?i)(" + "|".join(spices_list) + r")", key):
                    unit_replace = " g"
                    quantity_num = 0
                if re.search(r"(½|demi)", key):
                    quantity_num = 0.5
                    unit_replace = " piece"
                if re.search(r"\(([^\)]+)\)", key):
                    key = re.sub(r"\(([^\)]+)\)", "", key)
                if re.search(r"^[A-z].{0,1}(?= )", key):
                    key = re.sub(r"^[A-z].{0,1}(?= )", "", key)
                if re.search(r"((cac de)|(cac d'))", key):
                    key = re.sub(r"((cac de)|(cac d'))", "", key)
                    quantity_num = quantity_num * 0.005
                    unit_replace = " L"
                elif re.search(r"((cas de)|(cas d'))", key):
                    key = re.sub(r"((cas de)|(cas d'))", "", key)
                    quantity_num = quantity_num * 0.015
                    unit_replace = " L"
                if re.search(r"([\d\/.,]+)\s*", key):
                    key = re.sub(r"([\d\/.,]+)\s*", "", key)
                if re.search(r"^(ca cafe)", key):
                    key = re.sub(r"^(ca cafe)", "", key)
                elif re.search(
                    r"^(ca soupe|soupe|cuiller a soupe|cuillere a soupe )", key
                ):
                    key = re.sub(
                        r"^(ca soupe|soupe|cuiller a soupe|cuillere a soupe )",
                        "",
                        key,
                    )
                if re.search(r"^(s )", key):
                    key = re.sub(r"^(s )", "", key)
                if re.search(r"^(g )|^(G )", key):
                    key = re.sub(r"^(g )|^(G )", "", key)
                    unit_replace = " g"
                if re.search(r"^(l )|^(L )", key):
                    key = re.sub(r"^(l )|^(L )", "", key)
                    unit_replace = " g"
                if re.search(r"^(de )|^(d')", key):
                    key = re.sub(r"^(de )|^(d')", "", key)
                new_value += str(quantity_num) + unit_replace
            else:
                new_value += str(quantity_num) + " piece"

            new_dict[key.strip()] = new_value
        new_dict_list.append(new_dict)
    return new_dict_list
