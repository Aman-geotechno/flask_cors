from langchain_community.utilities import SQLDatabase

from sqlalchemy import create_engine
import warnings
warnings.filterwarnings("ignore")

from operator import itemgetter
from langchain.chains import create_sql_query_chain
from langchain_core.runnables import RunnablePassthrough
from langchain.chains.sql_database.prompt import PROMPT_SUFFIX
from langchain.prompts.prompt import PromptTemplate
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain.prompts import SemanticSimilarityExampleSelector
from langchain.prompts import FewShotPromptTemplate
from langchain.memory import ConversationBufferMemory
from langchain.chains.sql_database.prompt import _postgres_prompt


import os
from sqlalchemy import create_engine
from langchain.chains.openai_tools import create_extraction_chain_pydantic
from langchain_core.pydantic_v1 import BaseModel, Field
from langchain_openai import ChatOpenAI
from langchain_google_genai import GoogleGenerativeAI,ChatGoogleGenerativeAI
from langchain_community.vectorstores import FAISS
from prompted import examples
from flask import Flask, request, jsonify
from sqlalchemy.exc import SQLAlchemyError
from flask import Flask
from flask_cors import CORS
#from langchain_google_vertexai import ChatVertexAI

#import prompted
import psycopg2




from dotenv import load_dotenv
import os
load_dotenv()


os.environ["LANGCHAIN_TRACING_V2"] = "true"
os.environ["LANGCHAIN_PROJECT"] = "Jeevika_Traces"
os.environ["LANGCHAIN_ENDPOINT"] = "https://api.smith.langchain.com"
os.environ["LANGCHAIN_API_KEY"] = os.getenv("LANGCHAIN_API_KEY")  # Update to your API key

app = Flask(__name__)
#CORS(app)


db_user = 'postgres'
db_password = '1234'
db_host = '10.0.0.19'
db_port = '5432'
db_name = 'jeevika_data_warehouse'


try:
        
    connection = psycopg2.connect(
            database='jeevika_data_warehouse',
            user='postgres',
            password='1234',
            host='10.0.0.19',
            port='5432'
        )
    print("Successfully connected to the PostgreSQL database")
        
except (Exception, psycopg2.Error) as error:
        print("Error while connecting to PostgreSQL database:", error)


db = SQLDatabase.from_uri(f"postgresql://{db_user}:{db_password}@{db_host}:{db_port}/{db_name}",include_tables=["m_cbo","m_cbo_type","m_cbo_member","m_cbo_shg_member","t_cbo_appl_mapping","t_cbo_loan_register","m_designation","t_acc_voucher","t_bulk_bank_acc","m_farmer", "m_farmer_crop","m_farmer_crop_technology", "m_farmer_croptype", "m_farmer_land",
"m_farmer_pest_management", "mp_cbo_member","m_farmer_seed", "m_farmer_soil_management","t_mp_farmer_transaction_pest", "t_mp_farmer_transaction_soil","t_mp_trasaction_croptechnology","m_block",
"m_district","m_designation","m_village","m_panchayat","clf_masik_grading",
                    "vo_masik_grading",
                    "shg_masik_grading","profile_entry","t_expenditure_details","t_sell_grain","t_digital_banking","t_advisory_farmer_entry","t_agri_input","t_marketing_services","t_nursery_services","m_expenditure_type","m_chc_details","t_farmer_booking","t_chc_expenditure_details","t_freight_details","neera_selling","neera_collection","m_pg","pg_non_pg_memberes","m_clcdc","t_vidya_didi","t_learner_profile","mp_nursery_fy","t_sell_plant","t_payment_receive_details","t_expenditure_details","profile_entry_2","m_bankdataupload","m_agentnew","mp_pg_member","t_household_batch","g_member_mapping","g_goatry_distribution","m_dcs_profile","mp_member_dcs","mp_pond_fpg_mapping","batch_creation","t_sell_details","m_pond","t_sell_details","mp_matasya_sakhi_pond_mapping",
"mp_member_with_fpg_mapping","m_profile", "m_shg_hns_user_table", "t_training_of_cadre_and_pmt","m_user_profile","t_patient_info","mp_cbo_member_activity","m_intervention_activity","employer_window", "plan_job_fair_training", "candidates_profile", "is_letter_offered","sale_report","res_company","res_partner","res_company_17"])


llm = ChatGoogleGenerativeAI(model="gemini-1.5-pro-latest",convert_system_message_to_human=True,google_api_key=os.getenv('GOOGLE_API_KEY'), temperature=0)


example_prompt = PromptTemplate(
input_variables=["input", "sql_cmd", "result", "answer",],
template="\nQuestion: {input}\nSQLQuery: {sql_cmd}\nSQLResult: {result}\nAnswer: {answer}",
)



embeddings = HuggingFaceEmbeddings()

to_vectorize = [" ".join(example.values()) for example in examples]
print('t1')

vectorstore = Chroma.from_texts(to_vectorize, embeddings, metadatas=examples)
print('t2')
example_selector = SemanticSimilarityExampleSelector(
    vectorstore=vectorstore,
    k=1,
)



    
   










































       
       
                                                                                                     
 
















few_shot_prompt = FewShotPromptTemplate(
    example_selector=example_selector,
    example_prompt=example_prompt,
    prefix=_postgres_prompt+"and keep in mind its very very important that while generating sql query donot put ;(semicolon) in the end of query or any special character or brackets at begining or end only give sql query....... Donot get confused in code and id both are same for example if there is BLOCK_CODE or BLOCK_ID both are same similarly if there is DISTRICT_CODE or DISTRICT_ID both are same ,\
same goes for village_id and village_code and panchayat_id and panchayat_code and clf_id and clf_code.\
You cannot perform join on clf_id with village_id or block_id or block_code\
but you can perform join on village_id and village_code ,similar for district_id and district_code and block_id and block_code.\
    ",
    suffix=PROMPT_SUFFIX,
    input_variables=["input", "table_info", "top_k"], 
)


query_chain = create_sql_query_chain(llm, db,prompt=few_shot_prompt)

    

  




class Table(BaseModel):
        """Table in SQL database."""

        name: str = Field(description="Name of table in SQL database.")

table_names = "\n".join(db.get_usable_table_names())
system = f"""Return the names of ALL the SQL tables that MIGHT be relevant to the user question. \
The tables are:

    {table_names}

Remember to include ALL POTENTIALLY RELEVANT tables, even if you're not sure that they're needed."""

table_chain = create_extraction_chain_pydantic(Table, llm, system_message=system)



system = f"""Return the names of the SQL tables that are relevant to the user question. \
The tables are:

CBO
Farmer"""
category_chain = create_extraction_chain_pydantic(Table, llm, system_message=system)


from typing import List


def get_tables(categories: List[Table]) -> List[str]:
        tables = []
        for category in categories:
            if category.name == "CBO":
                tables.extend(
                [
                    "m_cbo",
                    "m_cbo_type",
                    "m_cbo_member",
                    "m_cbo_shg_member",
                    "t_cbo_appl_mapping",
                    "t_cbo_loan_register",
                    "t_acc_voucher",
                    "t_bulk_bank_acc",
                    "mp_cbo_member",
                    "m_block",
                    "m_district",
                    "m_designation",
                    "m_village",
                    "m_panchayat",
                    "m_designation",
                    "clf_masik_grading",
                    "vo_masik_grading",
                    "shg_masik_grading",
                    "profile_entry",
                    "t_expenditure_details",
                    "t_sell_grain",
                    "t_digital_banking",
                    "t_advisory_farmer_entry",
                    "t_agri_input",
                    "t_marketing_services",
                    "t_nursery_services",
                    "m_expenditure_type",
                    "m_chc_details",
                    "t_farmer_booking",
                    "t_chc_expenditure_details",
                    "t_freight_details",
                    "neera_selling","neera_collection","m_pg","pg_non_pg_memberes","m_clcdc","t_vidya_didi","t_learner_profile","mp_nursery_fy","t_sell_plant","t_payment_receive_details","t_expenditure_details","profile_entry_2","m_bankdataupload","m_agentnew","mp_pg_member","t_household_batch","g_member_mapping","g_goatry_distribution","m_dcs_profile","mp_member_dcs","mp_pond_fpg_mapping","batch_creation","t_sell_details","m_pond","t_sell_details","mp_matasya_sakhi_pond_mapping",
"mp_member_with_fpg_mapping","m_profile", "m_shg_hns_user_table", "t_training_of_cadre_and_pmt","m_user_profile","t_patient_info","mp_cbo_member_activity","m_intervention_activity"
                ]
            )
            elif category.name == "Farmer":
                tables.extend(["m_farmer", "m_farmer_crop","m_farmer_crop_technology", "m_farmer_croptype", "m_farmer_land",
                            "m_farmer_pest_management", "m_farmer_seed", "m_farmer_soil_management","t_farmer_transaction","t_mp_farmer_transaction_pest", "t_mp_farmer_transaction_soil","t_mp_trasaction_croptechnology"])
        return tables


table_chain = category_chain | get_tables
table_chain = {"input": itemgetter("question")} | table_chain

full_chain = RunnablePassthrough.assign(table_names_to_use=table_chain) | query_chain





@app.route('/', methods=['GET'])
def api_test():
     
     return "server is running"


@app.route('/response', methods=['POST'])
def api():
    try:
        data = request.get_json()
        question = data.get('question', '')
       

        try:
            query = full_chain.invoke(
            {"question": question })
            llm2= GoogleGenerativeAI(model="gemini-1.5-pro-latest",google_api_key=os.getenv('GOOGLE_API_KEY'), temperature=0)

        
            print(query)
        except Exception as e:
             print(e)

        print("-"*20)
        print()
        print()





                                                                    

        
        

                                                                                                
        
























        try:
            final_query=llm2(f"""This is a postgres sql query {query} which is going to be executed against this question {question}... 
            Your task is to Only return query, remove eveything from query at the begining or the end, don't return ```sql
                         
            For example: 
            If query generated is like this... 

            ```sql
            select count(distinct cbo_id) as total_cbo from m_cbo
            ``` 

            then you should return 

            select count(distinct cbo_id) as total_cbo from m_cbo 

            Another example:
            If query generated is like this... 

            ```sql
            SELECT COUNT(a.clf_id) AS clf_count
            FROM clf_masik_grading a
            WHERE a.year = 2023 and a.month_name = 'Dec' AND a.final_grade = 'A'

            then you should return 

            SELECT COUNT(a.clf_id) AS clf_count
            FROM clf_masik_grading a
            WHERE a.year = 2023 and a.month_name = 'Dec' AND a.final_grade = 'A'
            """)
        
            print(final_query)
        
            #print(final_query)
        except Exception as e:
             print(e)

        #print(db.run(final_query))
        
        response = llm2(f"""this is user question {question} and this is the answer {db.run(final_query)}....combine both to give a final answer...this is very important to include only this value {db.run(final_query)} in your answer \ 
                        your final answer must include word like district or block depending on whether the sql query that is this {final_query}...contains block_name or district_name...if it contains block_name then in final answer include word block...if it contains district_name then include district word in final answer...if it contains both block_name and district_name then include district word in final answer.

                        For example:
 question:clf count in daniawan
 sql_cmd:SELECT COUNT(DISTINCT c.cbo_id) AS clf_count 
                            FROM m_cbo c
                            INNER JOIN m_cbo_type t ON c.cbo_type_id = t.cbo_type_id
                            INNER JOIN m_block b ON c.block_id = b.block_id
                            WHERE upper(t.type_short_name) = 'CLF' AND upper(b.block_name) = 'DANIAWAN' AND c.record_status=1
answer:[(56,)]
final_answer:There are 56 clf in daniawan block.....you can clearly see in final answer i have used suffix block has been used with daniawan as in sql query the b.block_name was used...if it had been d.district_name then in final answer district should be used.

Another example:
question:total vo in begusarai
sql_cmd:SELECT COUNT(c.CBO_ID) AS total_vos
                            FROM m_cbo c
                            INNER JOIN m_cbo_type t ON c.CBO_TYPE_ID = t.CBO_TYPE_ID
                            INNER JOIN m_district d ON c.DISTRICT_ID = d.DISTRICT_ID
                            WHERE upper(t.TYPE_SHORT_NAME) = 'VO' AND upper(d.DISTRICT_NAME) = 'BEGUSARAI'
                            AND c.record_status=1
answer:[(2017,)]
final_answer:There are 2017 vo in begusarai district.....you can clearly see the suffix district has been used with begusarai as in sql query d.DISTRICT_NAME name was used
        .....answer in pointwise....your final answer must include the values of {db.run(final_query)} \
        
       

Please ensure the response format aligns with the provided instructions.
         
        Remember that vo means village organisation,shg means self help group,cbo means community based organisation and clf means cluster level federation  \
                    Pay attention to not add anything from your side in answer.. just give simple natural language answer including this value {db.run(final_query)}. 
                
         Also if the answer has two or more columns then retun the answr in tabular format.Only return first 50 \
            
            Now one also important thing that if user query that is this..{question}...contains any word in this..(ARWAL
MADHUBANI
JAMUI
NAWADA
ROHTAS
KATIHAR
DARBHANGA
SAMASTIPUR
LAKHISARAI
BUXAR
AURANGABAD
BEGUSARAI
KHAGARIA
GOPALGANJ
KISHANGANJ
VAISHALI
SUPAUL
SHEOHAR
MUNGER
ARARIA
BANKA
JEHANABAD
MADHEPURA
SIWAN
SHEIKHPURA)..then it can be either district or block....so we can identify whether it is district or block by looking at the sql query generated which is this..{final_query}..if it contains block_name and district_name then include word block and district in final answer else if there is district then include district in final answer and if only block then include only block in final answer... \ 
 For example:
 question:clf count in daniawan
 sql_cmd:SELECT COUNT(DISTINCT c.cbo_id) AS clf_count 
                            FROM m_cbo c
                            INNER JOIN m_cbo_type t ON c.cbo_type_id = t.cbo_type_id
                            INNER JOIN m_block b ON c.block_id = b.block_id
                            WHERE upper(t.type_short_name) = 'CLF' AND upper(b.block_name) = 'DANIAWAN' AND c.record_status=1
answer:[(56,)]
final_answer:There are 56 clf in daniawan block.....you can clearly see in final answer i have used suffix block has been used with daniawan as in sql_cmd the b.block_name was used...if it had been d.district_name then in final answer district should be used.

Another example:
question:total vo in begusarai
sql_cmd:SELECT COUNT(c.CBO_ID) AS total_vos
                            FROM m_cbo c
                            INNER JOIN m_cbo_type t ON c.CBO_TYPE_ID = t.CBO_TYPE_ID
                            INNER JOIN m_district d ON c.DISTRICT_ID = d.DISTRICT_ID
                            WHERE upper(t.TYPE_SHORT_NAME) = 'VO' AND upper(d.DISTRICT_NAME) = 'BEGUSARAI'
                            AND c.record_status=1
answer:[(2017,)]
final_answer:There are 2017 vo in begusarai district.....you can clearly see the suffix district has been used with begusarai as in sql_cmd d.DISTRICT_NAME name was used""")
        print(response)

        
        print(db.run(final_query))

        print("-"*20)

        cursor = connection.cursor()
       
            
        cursor.execute(final_query)

        res=jsonify(response)
        res.headers.add("Access-Control-Allow-Origin",'*')

    
    
    
       
        return res
    except SQLAlchemyError as e:
        return jsonify({'error': f'Database error: {str(e)}'}), 500
    except Exception as e:
        return jsonify({'error': str(e)}), 500
if __name__ == '__main__':
    
    app.run(host='0.0.0.0',port=5000)
