from dataclasses import dataclass


@dataclass
class Address:
    street: str
    house: str


ADDRESSES: list[Address] = [
    Address(street="Ленина", house="51"),
    Address(street="Мира", house="19"),
    Address(street="Мира", house="32"),
]


def get_address(street: str, house: str) -> Address:
    address = next(
        (
            address for address in ADDRESSES
            if address.street.strip().lower() == street.strip().lower()
            and address.house.strip().lower() == house.strip().lower()
        ),
        None
    )

    if address is None:
        address = Address(street=street, house=house)
        ADDRESSES.append(address)

    return address
