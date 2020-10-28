"""Glass AI ORMs

Assumptions:
- Organisation names and URL's do not change
- Sectors do not change TODO
- Addresses do not change TODO
"""
import re
from sqlalchemy import Table, MetaData, Column, Integer, String, ForeignKey, create_engine, Date, Boolean
from sqlalchemy.orm import Mapper, relationship, sessionmaker, backref
from research_daps import declarative_base

Base = declarative_base()

organisation_address = Table(
    "organisation_address",
    Base.metadata,
    Column("org_id", Integer, ForeignKey("organisation.org_id")),
    Column("address_id", Integer, ForeignKey("address.address_id")),
    # Active?
    # Rank
    # Date
)

organisation_sector = Table(
    "organisation_sector",
    Base.metadata,
    Column("org_id", Integer, ForeignKey("organisation.org_id")),
    Column("sector_id", Integer, ForeignKey("sector.sector_id")),
    # Active?
)

notice_terms = Table(
    "notice_terms",
    Base.metadata,
    Column("notice_id", Integer, ForeignKey("notice.notice_id")),
    Column("term_id", Integer, ForeignKey("covid_term.term_id")),
)

class Organisation(Base):
    """An organisation relating to a business website"""
    org_id = Column(Integer, primary_key=True)
    name = Column(String, unique=True, doc="Organisation name inferred by named entity recognition", index=True)
    website = Column(String, nullable=False, doc="URL of organisation")
    active = Column(Boolean, nullable=False, doc="True if Organisation was contained in last data-dump")
    # One-to-many
    notices = relationship(
        "Notice", backref=backref("organisation")
    )
    descriptions = relationship(
        "OrganisationDescription", backref=backref("organisation")
    )
    metadatas = relationship("OrganisationMetadata", backref=backref("organisation"))
    # Many-to-many
    sectors = relationship("Sector", secondary=organisation_sector, back_populates="organisations")
    addresses = relationship("Address", secondary=organisation_address, back_populates="organisations")

    def __repr__(self):
        return f"<Org(name={self.name})>"

class OrganisationMetadata(Base):
    """Organisation metadata which may not be stable over time"""
    org_id = Column(Integer, ForeignKey("organisation.org_id"), primary_key=True)
    date = Column(Date, primary_key=True, doc="Date of data-dump inserting row", index=True)
    has_webshop = Column(Boolean, nullable=False, doc="If True, presence of a webshop was found")
    vat_number = Column(String, doc="VAT number (low population)")
    low_quality = Column(Boolean, nullable=False, doc="If True, information for `org_id` was of low quality")


class OrganisationDescription(Base):
    """Descriptions of organisation activities"""
    # description_id = Column(Integer, primary_key=True)
    org_id = Column(Integer, ForeignKey("organisation.org_id"))
    description = Column(String, nullable=False, doc="Description of organisation extracted from website")
    date = Column(Date, nullable=False, doc="Date of data-dump inserting row", index=True)  # TODO this should be a secondary key


class Address(Base):
    """List of addresses found in websites"""
    address_id = Column(Integer, primary_key=True)
    address_text = Column(String, nullable=False, unique=True, doc="Full address text")
    postcode = Column(String, doc="Postcode of address")
    organisations = relationship("Organisation", secondary=organisation_address, back_populates="addresses")

class Notice(Base):
    """Covid Notices extracted from websites"""
    notice_id = Column(Integer, primary_key=True)
    org_id = Column("org_id", Integer, ForeignKey("organisation.org_id"))
    snippet = Column(String, nullable=False, doc="Extracted text snippet relating to COVID")
    url = Column(String, nullable=False, doc="URL snippet was extracted from")
    date = Column(Date, nullable=False, doc="Date of data-dump inserting row", index=True)
    terms = relationship("CovidTerm", secondary=notice_terms, back_populates="notices")

    def __repr__(self):
        return f"<Notice(url={self.url})>"

class CovidTerm(Base):
    """Set of terms relating to Covid-19, curated by Glass.AI
    Terms are used to find notices
    """
    term_id = Column(Integer, primary_key=True)
    term_string = Column(String, unique=True, index=True)
    date_introduced = Column(Date, doc="Date of data-dump term was first used to find notices")
    notices = relationship("Notice", secondary=notice_terms, back_populates="terms")

class Sector(Base):
    """Sector names: LinkedIn taxonomy"""
    sector_id = Column(Integer, primary_key=True)
    sector_name = Column(String, unique=True, doc="Name of sector")
    organisations = relationship("Organisation", secondary=organisation_sector, back_populates="sectors")


# # Just a table?
# class OrganisationCompaniesHouseMatch(Base):
#     """Organisation matches to companies house performed by Glass"""
#     glass_match_id = Column(Integer, primary_key=True)
#     company_id = Column(String, ForeignKey("companies.company_id"), doc="Companies House number")
#     org_id = Column("org_id", Integer, ForeignKey("organisation.org_id"))
#     company_match_type = Column(String, doc="Type of match: MATCH_3,MATCH_4,MATCH_5")

try:
    o = Organisation(name='foo', website="bar", addresses=[Address(address_text="a")])
except Exception as e:
    print(e)

    exit()
