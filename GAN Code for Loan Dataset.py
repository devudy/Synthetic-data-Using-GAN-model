{
  "cells": [
    {
      "cell_type": "markdown",
      "source": [
        "## Code for Loans Dataset\n",
        "\n",
        "---\n",
        "\n"
      ],
      "metadata": {
        "id": "5XYe9Ptvg2vp"
      },
      "id": "5XYe9Ptvg2vp"
    },
    {
      "cell_type": "markdown",
      "source": [
        "*Note: due to problems arising when trying to use pandas-profiling, Visual and Statistical evaluation of dataset were done manually.*"
      ],
      "metadata": {
        "id": "vGKsRdQrg9ma"
      },
      "id": "vGKsRdQrg9ma"
    },
    {
      "cell_type": "code",
      "execution_count": null,
      "metadata": {
        "id": "636a9af1",
        "outputId": "0c3586fc-b8ce-46a4-8203-6ddf8790a480"
      },
      "outputs": [
        {
          "name": "stdout",
          "output_type": "stream",
          "text": [
            "/Users/apple/Desktop/NayaOneProject\n"
          ]
        }
      ],
      "source": [
        "import os\n",
        "\n",
        "HOME = os.path.expanduser('~')\n",
        "PROJECT_DIR = os.path.join(HOME, 'Desktop', 'NayaOneProject')\n",
        "os.chdir(PROJECT_DIR)\n",
        "print(os.getcwd())"
      ],
      "id": "636a9af1"
    },
    {
      "cell_type": "code",
      "execution_count": null,
      "metadata": {
        "id": "a8660837"
      },
      "outputs": [],
      "source": [
        "#Import Necessary Libraries"
      ],
      "id": "a8660837"
    },
    {
      "cell_type": "code",
      "source": [
        "#Libraries for EDA"
      ],
      "metadata": {
        "id": "LHLGBfCpk6rn"
      },
      "id": "LHLGBfCpk6rn",
      "execution_count": null,
      "outputs": []
    },
    {
      "cell_type": "code",
      "execution_count": null,
      "metadata": {
        "id": "a42d54bd"
      },
      "outputs": [],
      "source": [
        "import pandas as pd\n",
        "import seaborn as sns\n",
        "import matplotlib.pyplot as plt"
      ],
      "id": "a42d54bd"
    },
    {
      "cell_type": "code",
      "source": [
        "#Libraries for CTGAN"
      ],
      "metadata": {
        "id": "SYdQ4yZYlCXw"
      },
      "id": "SYdQ4yZYlCXw",
      "execution_count": null,
      "outputs": []
    },
    {
      "cell_type": "code",
      "execution_count": null,
      "metadata": {
        "id": "TpFP2_Wb3Wbr"
      },
      "outputs": [],
      "source": [
        "from sdv.tabular import CTGAN\n",
        "from sdv.constraints import Inequality"
      ],
      "id": "TpFP2_Wb3Wbr"
    },
    {
      "cell_type": "code",
      "source": [
        "#Libraries necessary for evaluation"
      ],
      "metadata": {
        "id": "7mKWU4kKl9Jf"
      },
      "id": "7mKWU4kKl9Jf",
      "execution_count": null,
      "outputs": []
    },
    {
      "cell_type": "code",
      "execution_count": null,
      "metadata": {
        "id": "0_8T8hBV6V3e"
      },
      "outputs": [],
      "source": [
        "from sdv.evaluation import evaluate\n",
        "from table-evaluator import TableEvaluator"
      ],
      "id": "0_8T8hBV6V3e"
    },
    {
      "cell_type": "code",
      "source": [
        "#Import Dataset"
      ],
      "metadata": {
        "id": "Ms6BndQWdM6L"
      },
      "id": "Ms6BndQWdM6L",
      "execution_count": null,
      "outputs": []
    },
    {
      "cell_type": "code",
      "execution_count": null,
      "metadata": {
        "id": "5cef3fd1",
        "outputId": "78153508-0469-4345-c9d2-9a932755b2fa"
      },
      "outputs": [
        {
          "name": "stderr",
          "output_type": "stream",
          "text": [
            "/Users/apple/opt/anaconda3/lib/python3.8/site-packages/IPython/core/interactiveshell.py:3165: DtypeWarning: Columns (19,55) have mixed types.Specify dtype option on import or set low_memory=False.\n",
            "  has_raised = await self.run_ast_nodes(code_ast.body, cell_name,\n"
          ]
        }
      ],
      "source": [
        "Loans = pd.read_csv(\"lc_loan.csv\")"
      ],
      "id": "5cef3fd1"
    },
    {
      "cell_type": "code",
      "source": [
        "#Inspect Dataset"
      ],
      "metadata": {
        "id": "j8dXlpiBdQiZ"
      },
      "id": "j8dXlpiBdQiZ",
      "execution_count": null,
      "outputs": []
    },
    {
      "cell_type": "code",
      "source": [
        "Loans.info()"
      ],
      "metadata": {
        "id": "XezXDthUmhFA"
      },
      "id": "XezXDthUmhFA",
      "execution_count": null,
      "outputs": []
    },
    {
      "cell_type": "code",
      "execution_count": null,
      "metadata": {
        "id": "5e1ac2db"
      },
      "outputs": [],
      "source": [
        "#Drop Irrelevant Columns and Columns Missing a large portion of their values. \n",
        "#Or that were observed to decrease accuracy of CTGAN in early fitments."
      ],
      "id": "5e1ac2db"
    },
    {
      "cell_type": "code",
      "execution_count": null,
      "metadata": {
        "id": "2526aa0f"
      },
      "outputs": [],
      "source": [
        "Loans = Loans.drop(columns=['url','id','member_id','zip_code','grade','emp_title','title','issue_d','annual_inc_joint','dti_joint','verification_status_joint','last_pymnt_d','next_pymnt_d','last_credit_pull_d','earliest_cr_line','desc','policy_code'])"
      ],
      "id": "2526aa0f"
    },
    {
      "cell_type": "code",
      "execution_count": null,
      "metadata": {
        "id": "f65f17c6"
      },
      "outputs": [],
      "source": [
        "Loans.info()"
      ],
      "id": "f65f17c6"
    },
    {
      "cell_type": "code",
      "source": [
        "#Drop columns with large number of missing values"
      ],
      "metadata": {
        "id": "aL6ZSquomow0"
      },
      "id": "aL6ZSquomow0",
      "execution_count": null,
      "outputs": []
    },
    {
      "cell_type": "code",
      "execution_count": null,
      "metadata": {
        "id": "99b550b6"
      },
      "outputs": [],
      "source": [
        "Loans.drop(Loans.iloc[:,42:53],axis=1,inplace=True)"
      ],
      "id": "99b550b6"
    },
    {
      "cell_type": "code",
      "execution_count": null,
      "metadata": {
        "id": "b264074e"
      },
      "outputs": [],
      "source": [
        "Loans.drop(Loans.iloc[:,43:],axis=1,inplace=True)"
      ],
      "id": "b264074e"
    },
    {
      "cell_type": "code",
      "execution_count": null,
      "metadata": {
        "id": "5c438b79"
      },
      "outputs": [],
      "source": [
        "#Missing Value imputation for Month Variables:"
      ],
      "id": "5c438b79"
    },
    {
      "cell_type": "code",
      "execution_count": null,
      "metadata": {
        "id": "3be10688"
      },
      "outputs": [],
      "source": [
        "Loans.mths_since_last_major_derog.describe()"
      ],
      "id": "3be10688"
    },
    {
      "cell_type": "code",
      "source": [
        "#Assign value outside range of values in above output, i.e. a value less than \n",
        "#the minimum to all missing values. This will result in only the null entries \n",
        "#of the column to be filled with a made up number -1. "
      ],
      "metadata": {
        "id": "vgSE81YjbXtG"
      },
      "id": "vgSE81YjbXtG",
      "execution_count": null,
      "outputs": []
    },
    {
      "cell_type": "code",
      "execution_count": null,
      "metadata": {
        "id": "a5a9067b"
      },
      "outputs": [],
      "source": [
        "Loans = Loans.fillna(value={'mths_since_last_major_derog':-1})"
      ],
      "id": "a5a9067b"
    },
    {
      "cell_type": "code",
      "source": [
        "#Assign ranges to number of months and No Major Derog to -1, resulting in the null values \n",
        "#to be imputed correctly as No Major derog, since initially # of months is not present\n",
        "#in the column"
      ],
      "metadata": {
        "id": "cGk1anLPb2zK"
      },
      "id": "cGk1anLPb2zK",
      "execution_count": null,
      "outputs": []
    },
    {
      "cell_type": "code",
      "execution_count": null,
      "metadata": {
        "id": "663096df"
      },
      "outputs": [],
      "source": [
        "Loans['Last_Major_Derog_Length']=pd.cut(Loans.mths_since_last_major_derog,bins=[-1.5,-0.5,6,12,24,48,96,200],labels=['No Major Derog','<6months','<12months','1-2 years','2-4 years','4-8years','8+ years'])"
      ],
      "id": "663096df"
    },
    {
      "cell_type": "code",
      "source": [
        "#Check if imputation worked as intendeded by referencing summary statistics above."
      ],
      "metadata": {
        "id": "Ms_p3lgMY8O_"
      },
      "id": "Ms_p3lgMY8O_",
      "execution_count": null,
      "outputs": []
    },
    {
      "cell_type": "code",
      "execution_count": null,
      "metadata": {
        "id": "d4316cdd"
      },
      "outputs": [],
      "source": [
        "Loans.Last_Major_Derog_Length.value_counts()"
      ],
      "id": "d4316cdd"
    },
    {
      "cell_type": "code",
      "execution_count": null,
      "metadata": {
        "id": "546140ce"
      },
      "outputs": [],
      "source": [
        "Loans.mths_since_last_delinq.describe()"
      ],
      "id": "546140ce"
    },
    {
      "cell_type": "code",
      "source": [
        "#Assign value outside range of values in above output, i.e. a value less than \n",
        "#the minimum to all missing values. This will result in only the null entries \n",
        "#of the column to be filled with a made up number -1. "
      ],
      "metadata": {
        "id": "J5XUylYdcK-x"
      },
      "id": "J5XUylYdcK-x",
      "execution_count": null,
      "outputs": []
    },
    {
      "cell_type": "code",
      "execution_count": null,
      "metadata": {
        "id": "1d6281a4"
      },
      "outputs": [],
      "source": [
        "Loans = Loans.fillna(value={'mths_since_last_delinq':-1})"
      ],
      "id": "1d6281a4"
    },
    {
      "cell_type": "code",
      "source": [
        "#Assign ranges to number of monthsand No Delinq to -1, resulting in the null values to \n",
        "#be imputed correctly as No Delinq, since initially # of months is not present\n",
        "#in the column"
      ],
      "metadata": {
        "id": "3WvYhI1FcLbK"
      },
      "id": "3WvYhI1FcLbK",
      "execution_count": null,
      "outputs": []
    },
    {
      "cell_type": "code",
      "execution_count": null,
      "metadata": {
        "id": "861601e1"
      },
      "outputs": [],
      "source": [
        "Loans['Last_Delinq_Length']=pd.cut(Loans.mths_since_last_delinq ,bins=[-1.5,-0.5,6,12,24,48,96,200],labels=['No Delinq','<6months','<12months','1-2 years','2-4 years','4-8years','8+ years'])"
      ],
      "id": "861601e1"
    },
    {
      "cell_type": "code",
      "source": [
        "#Check if imputation worked as intendeded by referencing summary statistics above."
      ],
      "metadata": {
        "id": "W73xeu83Y6o_"
      },
      "id": "W73xeu83Y6o_",
      "execution_count": null,
      "outputs": []
    },
    {
      "cell_type": "code",
      "execution_count": null,
      "metadata": {
        "id": "33b590af"
      },
      "outputs": [],
      "source": [
        "Loans.Last_Delinq_Length.value_counts()"
      ],
      "id": "33b590af"
    },
    {
      "cell_type": "code",
      "execution_count": null,
      "metadata": {
        "id": "88a30c1f"
      },
      "outputs": [],
      "source": [
        "Loans.mths_since_last_record.describe()"
      ],
      "id": "88a30c1f"
    },
    {
      "cell_type": "code",
      "source": [
        "#Assign value outside range of values in above output, i.e. a value less than \n",
        "#the minimum to all missing values. This will result in only the null entries \n",
        "#of the column to be filled with a made up number -1. "
      ],
      "metadata": {
        "id": "VSo992wocPAG"
      },
      "id": "VSo992wocPAG",
      "execution_count": null,
      "outputs": []
    },
    {
      "cell_type": "code",
      "execution_count": null,
      "metadata": {
        "id": "94849789"
      },
      "outputs": [],
      "source": [
        "Loans = Loans.fillna(value={'mths_since_last_record':-1})"
      ],
      "id": "94849789"
    },
    {
      "cell_type": "code",
      "source": [
        "#Assign ranges to number of months and No Previous Record to -1, \n",
        "#resulting in the null values to  \n",
        "#be imputed correctly as No Previous Record, since initially # of months is not present\n",
        "#in the column"
      ],
      "metadata": {
        "id": "ga9Zrqimcazu"
      },
      "id": "ga9Zrqimcazu",
      "execution_count": null,
      "outputs": []
    },
    {
      "cell_type": "code",
      "execution_count": null,
      "metadata": {
        "id": "e12aaa9d"
      },
      "outputs": [],
      "source": [
        "Loans['Last_Record_Length']=pd.cut(Loans.mths_since_last_record ,bins=[-1.5,-0.5,6,12,24,48,96,200],labels=['No Previous Record','<6months','<12months','1-2 years','2-4 years','4-8years','8+ years'])"
      ],
      "id": "e12aaa9d"
    },
    {
      "cell_type": "code",
      "source": [
        "#Check if imputation worked as intendeded by referencing summary statistics above."
      ],
      "metadata": {
        "id": "iADL24_UYvjw"
      },
      "id": "iADL24_UYvjw",
      "execution_count": null,
      "outputs": []
    },
    {
      "cell_type": "code",
      "execution_count": null,
      "metadata": {
        "id": "6227011f"
      },
      "outputs": [],
      "source": [
        "Loans.Last_Record_Length.value_counts()"
      ],
      "id": "6227011f"
    },
    {
      "cell_type": "code",
      "execution_count": null,
      "metadata": {
        "id": "d1a9a320"
      },
      "outputs": [],
      "source": [
        "#Drop original Months Columns"
      ],
      "id": "d1a9a320"
    },
    {
      "cell_type": "code",
      "execution_count": null,
      "metadata": {
        "id": "550faa55"
      },
      "outputs": [],
      "source": [
        "Loans = Loans.drop(columns=['mths_since_last_major_derog','mths_since_last_record','mths_since_last_delinq'])"
      ],
      "id": "550faa55"
    },
    {
      "cell_type": "code",
      "execution_count": null,
      "metadata": {
        "id": "5df8ec48"
      },
      "outputs": [],
      "source": [
        "#Encode Address state into larger Regions in the US:"
      ],
      "id": "5df8ec48"
    },
    {
      "cell_type": "code",
      "execution_count": null,
      "metadata": {
        "id": "cc37569e"
      },
      "outputs": [],
      "source": [
        "Loans.addr_state.value_counts()"
      ],
      "id": "cc37569e"
    },
    {
      "cell_type": "code",
      "execution_count": null,
      "metadata": {
        "id": "79a78cf5"
      },
      "outputs": [],
      "source": [
        "#changing from states to regions\n",
        "#West Region: Pacific & Mountain\n",
        "Loans=Loans.replace(to_replace=['CA','OR','WA','HI','AK'], value='Pacific')\n",
        "Loans=Loans.replace(to_replace=['NV','ID','MT','WY','UT','CO','AZ','NM'], value='Mountain')\n",
        "#MidWest Region: West NorthCentral & East NorthCentral\n",
        "Loans=Loans.replace(to_replace=['WI','IL','MI','IN','OH'], value='East North Central')\n",
        "Loans=Loans.replace(to_replace=['ND','SD','NE','KS','MN','IA','MO'], value='West North Cetral')\n",
        "#North-East Region:Middle Atlantic & New England\n",
        "Loans=Loans.replace(to_replace=['NY','ME','CT','RI','NJ'], value='Middle Atlantic')\n",
        "Loans=Loans.replace(to_replace=['PA','VT','ME','NH','MA'], value='New England')\n",
        "#South Region: West South Central, East South Central, South Atlantic\n",
        "Loans=Loans.replace(to_replace=['TX','OK','AR','LA'], value='West South Central')\n",
        "Loans=Loans.replace(to_replace=['KY','TN','MS','AL'], value='East South Central')\n",
        "Loans=Loans.replace(to_replace=['DE','MD','DC','WV','VA','NC','SC','GA','FL'], value='South Atlantic')"
      ],
      "id": "79a78cf5"
    },
    {
      "cell_type": "code",
      "execution_count": null,
      "metadata": {
        "id": "44e2c300"
      },
      "outputs": [],
      "source": [
        "#Drop Remaining Null Values:"
      ],
      "id": "44e2c300"
    },
    {
      "cell_type": "code",
      "execution_count": null,
      "metadata": {
        "id": "ed3d06e8"
      },
      "outputs": [],
      "source": [
        "Loans = Loans.dropna()"
      ],
      "id": "ed3d06e8"
    },
    {
      "cell_type": "code",
      "execution_count": null,
      "metadata": {
        "id": "cf7143d8"
      },
      "outputs": [],
      "source": [
        "#Turn Object Columns to Categorical:"
      ],
      "id": "cf7143d8"
    },
    {
      "cell_type": "code",
      "execution_count": null,
      "metadata": {
        "id": "65ccc6d6"
      },
      "outputs": [],
      "source": [
        "Loans[Loans.select_dtypes(['object']).columns] = Loans.select_dtypes(['object']).apply(lambda x: x.astype('category'))"
      ],
      "id": "65ccc6d6"
    },
    {
      "cell_type": "code",
      "execution_count": null,
      "metadata": {
        "id": "ac4992c5"
      },
      "outputs": [],
      "source": [
        "#Key Summary Statistics:"
      ],
      "id": "ac4992c5"
    },
    {
      "cell_type": "code",
      "execution_count": null,
      "metadata": {
        "id": "e9ac3d49"
      },
      "outputs": [],
      "source": [
        "#Numerical Variable Statistics:"
      ],
      "id": "e9ac3d49"
    },
    {
      "cell_type": "code",
      "execution_count": null,
      "metadata": {
        "id": "bdde7870"
      },
      "outputs": [],
      "source": [
        "Loans[['loan_amnt','installment','int_rate','annual_inc','total_pymnt']].describe().transpose()"
      ],
      "id": "bdde7870"
    },
    {
      "cell_type": "code",
      "execution_count": null,
      "metadata": {
        "id": "a69294be"
      },
      "outputs": [],
      "source": [
        "#Categorical Variable Statistics:"
      ],
      "id": "a69294be"
    },
    {
      "cell_type": "code",
      "execution_count": null,
      "metadata": {
        "id": "42182175"
      },
      "outputs": [],
      "source": [
        "Loans[['emp_length','verification_status','loan_status','addr_state','home_ownership']].describe(include=object).transpose()"
      ],
      "id": "42182175"
    },
    {
      "cell_type": "code",
      "execution_count": null,
      "metadata": {
        "id": "3b359b65"
      },
      "outputs": [],
      "source": [
        "#Correlation Matrix Of Numerical Variables"
      ],
      "id": "3b359b65"
    },
    {
      "cell_type": "code",
      "execution_count": null,
      "metadata": {
        "id": "0cb6a456"
      },
      "outputs": [],
      "source": [
        "corrmatrix = Loans.corr()\n",
        "plt.figure(figsize=(15, 12))\n",
        "sns.heatmap(corrmatrix, annot=False)\n",
        "plt.savefig('Correlation Matrix.png')"
      ],
      "id": "0cb6a456"
    },
    {
      "cell_type": "code",
      "execution_count": null,
      "metadata": {
        "id": "dd582693"
      },
      "outputs": [],
      "source": [
        "#Graphs Of distribution Plots"
      ],
      "id": "dd582693"
    },
    {
      "cell_type": "code",
      "execution_count": null,
      "metadata": {
        "id": "1fbc8ff7"
      },
      "outputs": [],
      "source": [
        "sns.displot(Loans, x=\"loan_amnt\", kind=\"kde\").set(title='Original Distribution of Loan Amount',xlabel='Loan Amount')"
      ],
      "id": "1fbc8ff7"
    },
    {
      "cell_type": "code",
      "execution_count": null,
      "metadata": {
        "id": "8b66ae57"
      },
      "outputs": [],
      "source": [
        "sns.displot(Loans, x=\"total_pymnt\", kind=\"kde\").set(title='Original Distribution of Total Payment',xlabel='Total Payment')"
      ],
      "id": "8b66ae57"
    },
    {
      "cell_type": "code",
      "execution_count": null,
      "metadata": {
        "id": "818aa3b9"
      },
      "outputs": [],
      "source": [
        "sns.displot(Loans, x=\"installment\", kind=\"kde\").set(title='Original Distribution of Installment',xlabel='installment')"
      ],
      "id": "818aa3b9"
    },
    {
      "cell_type": "code",
      "execution_count": null,
      "metadata": {
        "id": "b43189da"
      },
      "outputs": [],
      "source": [
        "LoanStatusDist = sns.catplot(y=\"loan_status\",kind=\"count\", data=Loans).set(title='Original Distribution of Loan Status',ylabel='Status Of Loan')"
      ],
      "id": "b43189da"
    },
    {
      "cell_type": "code",
      "execution_count": null,
      "metadata": {
        "id": "0a922add"
      },
      "outputs": [],
      "source": [
        "LoanStatusDist = sns.catplot(y=\"home_ownership\",kind=\"count\", data=Loans).set(title='Original Distribution of Home Ownership',ylabel='Home Ownership')"
      ],
      "id": "0a922add"
    },
    {
      "cell_type": "code",
      "execution_count": null,
      "metadata": {
        "id": "bda08d5b"
      },
      "outputs": [],
      "source": [
        "LoanStatusDist = sns.catplot(y=\"addr_state\",kind=\"count\", data=Loans).set(title='Original Distribution of Areas of Loans Issued',ylabel='Areas in the US')"
      ],
      "id": "bda08d5b"
    },
    {
      "cell_type": "code",
      "execution_count": null,
      "metadata": {
        "id": "e22c3aa2"
      },
      "outputs": [],
      "source": [
        "ppdata = Loans[['loan_amnt', 'installment', 'int_rate','loan_status','total_pymnt']].copy()"
      ],
      "id": "e22c3aa2"
    },
    {
      "cell_type": "code",
      "execution_count": null,
      "metadata": {
        "id": "0d37ca04"
      },
      "outputs": [],
      "source": [
        "sns.pairplot(ppdata)"
      ],
      "id": "0d37ca04"
    },
    {
      "cell_type": "code",
      "source": [
        "plt.scatter(Loans.loan_amnt,Loans.installment)\n",
        "plt.title(\"Loan Amount VS Installment\")\n",
        "plt.xlabel(\"Loan Amount\")\n",
        "plt.ylabel(\"Installment\")\n",
        "plt.savefig('Corr1.png')\n",
        "plt.show()"
      ],
      "metadata": {
        "id": "pCptCJYcRw_W"
      },
      "id": "pCptCJYcRw_W",
      "execution_count": null,
      "outputs": []
    },
    {
      "cell_type": "code",
      "source": [
        "plt.scatter(Loans.loan_amnt,Loans.funded_amnt)\n",
        "plt.title(\"Loan Amount VS Funded Amount\")\n",
        "plt.xlabel(\"Loan Amount\")\n",
        "plt.ylabel(\"Funded Amount\")\n",
        "plt.savefig('Corr2.png')\n",
        "plt.show()"
      ],
      "metadata": {
        "id": "ijg18i9XRzPD"
      },
      "id": "ijg18i9XRzPD",
      "execution_count": null,
      "outputs": []
    },
    {
      "cell_type": "code",
      "source": [
        "samplecorrmatrix = Loans.corr()\n",
        "plt.figure(figsize=(15, 12))\n",
        "sns.heatmap(samplecorrmatrix, annot=False)\n",
        "plt.savefig('Original Correlation Matrix.png')"
      ],
      "metadata": {
        "id": "_Gcde3O7ndmg"
      },
      "id": "_Gcde3O7ndmg",
      "execution_count": null,
      "outputs": []
    },
    {
      "cell_type": "code",
      "execution_count": null,
      "metadata": {
        "id": "db997eaa"
      },
      "outputs": [],
      "source": [
        "#Create Samples out of Processed Dataset Using simple random sample and two coefficients,\n",
        "#0.1 and 0.01"
      ],
      "id": "db997eaa"
    },
    {
      "cell_type": "code",
      "execution_count": null,
      "metadata": {
        "id": "76910d8a"
      },
      "outputs": [],
      "source": [
        "SampleLoans = Loans.sample(frac = 0.1)"
      ],
      "id": "76910d8a"
    },
    {
      "cell_type": "code",
      "source": [
        "SmallSample = Loans.sample(frac=0.01)"
      ],
      "metadata": {
        "id": "3hqF8zaCuk5f"
      },
      "id": "3hqF8zaCuk5f",
      "execution_count": null,
      "outputs": []
    },
    {
      "cell_type": "code",
      "execution_count": null,
      "metadata": {
        "id": "1nHuLZGH-Cu3"
      },
      "outputs": [],
      "source": [
        "SampleLoans.info()"
      ],
      "id": "1nHuLZGH-Cu3"
    },
    {
      "cell_type": "code",
      "execution_count": null,
      "metadata": {
        "id": "a35a7567"
      },
      "outputs": [],
      "source": [
        "#Corr Matrix of Variables to see if original Correlations hold, and sampling hasn't \n",
        "#deteriorated originality of data."
      ],
      "id": "a35a7567"
    },
    {
      "cell_type": "code",
      "execution_count": null,
      "metadata": {
        "id": "ad472678"
      },
      "outputs": [],
      "source": [
        "samplecorrmatrix = SampleLoans.corr()\n",
        "plt.figure(figsize=(15, 12))\n",
        "sns.heatmap(samplecorrmatrix, annot=False)\n",
        "plt.savefig('Sample Correlation Matrix.png')"
      ],
      "id": "ad472678"
    },
    {
      "cell_type": "code",
      "execution_count": null,
      "metadata": {
        "id": "30023320"
      },
      "outputs": [],
      "source": [
        "#Introduce Constrains needed after original correlation inspection:"
      ],
      "id": "30023320"
    },
    {
      "cell_type": "code",
      "execution_count": null,
      "metadata": {
        "id": "3b81b7a6"
      },
      "outputs": [],
      "source": [
        "equality_constraint1 = Inequality(\n",
        "    low_column_name='funded_amnt',\n",
        "    high_column_name='loan_amnt'\n",
        ")"
      ],
      "id": "3b81b7a6"
    },
    {
      "cell_type": "code",
      "execution_count": null,
      "metadata": {
        "id": "9d29ad7b"
      },
      "outputs": [],
      "source": [
        "equality_constraint2 = Inequality(\n",
        "    low_column_name='funded_amnt_inv',\n",
        "    high_column_name='loan_amnt'\n",
        ")"
      ],
      "id": "9d29ad7b"
    },
    {
      "cell_type": "code",
      "execution_count": null,
      "metadata": {
        "id": "a358cd23"
      },
      "outputs": [],
      "source": [
        "inequality_constraint1 = Inequality(\n",
        "    low_column_name='funded_amnt_inv',\n",
        "    high_column_name='funded_amnt'\n",
        ")"
      ],
      "id": "a358cd23"
    },
    {
      "cell_type": "code",
      "execution_count": null,
      "metadata": {
        "id": "1ca816ad"
      },
      "outputs": [],
      "source": [
        "constraints = [equality_constraint1, equality_constraint2]"
      ],
      "id": "1ca816ad"
    },
    {
      "cell_type": "code",
      "execution_count": null,
      "metadata": {
        "id": "e0eec438"
      },
      "outputs": [],
      "source": [
        "constraints2 = [inequality_constraint1, equality_constraint1, equality_constraint2]"
      ],
      "id": "e0eec438"
    },
    {
      "cell_type": "code",
      "execution_count": null,
      "metadata": {
        "id": "d9300779"
      },
      "outputs": [],
      "source": [
        "# Define Different models to work with (please note numbers of models in the code do not align with numbers of models in the report. In particular, Model 1 in the report is model3 below, model 3 in the report is model1 below)"
      ],
      "id": "d9300779"
    },
    {
      "cell_type": "code",
      "execution_count": null,
      "metadata": {
        "id": "909e17bd"
      },
      "outputs": [],
      "source": [
        "model = CTGAN(constraints=constraints)"
      ],
      "id": "909e17bd"
    },
    {
      "cell_type": "code",
      "execution_count": null,
      "metadata": {
        "id": "31ee49d1"
      },
      "outputs": [],
      "source": [
        "model1 = CTGAN(constraints=constraints2, epochs=200, verbose = 'TRUE')"
      ],
      "id": "31ee49d1"
    },
    {
      "cell_type": "code",
      "execution_count": null,
      "metadata": {
        "id": "44f9c5ac"
      },
      "outputs": [],
      "source": [
        "model2 = CTGAN(constraints=constraints2, epochs=100, \n",
        "              batch_size=100, verbose='TRUE')"
      ],
      "id": "44f9c5ac"
    },
    {
      "cell_type": "code",
      "execution_count": null,
      "metadata": {
        "id": "aab01502"
      },
      "outputs": [],
      "source": [
        "model3 = CTGAN(verbose='TRUE')"
      ],
      "id": "aab01502"
    },
    {
      "cell_type": "code",
      "execution_count": null,
      "metadata": {
        "id": "f6ddae6e"
      },
      "outputs": [],
      "source": [
        "model4 = CTGAN(constraints=constraints)"
      ],
      "id": "f6ddae6e"
    },
    {
      "cell_type": "code",
      "execution_count": null,
      "metadata": {
        "id": "f1554c78"
      },
      "outputs": [],
      "source": [
        "model5 = CTGAN(constraints=constraints2, epochs=200, verbose='TRUE')"
      ],
      "id": "f1554c78"
    },
    {
      "cell_type": "code",
      "execution_count": null,
      "metadata": {
        "id": "00247279"
      },
      "outputs": [],
      "source": [
        "model6 = CTGAN(constraints=constraints2, epochs=200, batch_size = 100, verbose='TRUE')"
      ],
      "id": "00247279"
    },
    {
      "cell_type": "code",
      "execution_count": null,
      "metadata": {
        "id": "812768da"
      },
      "outputs": [],
      "source": [
        "#Fit models to original sample as fitting to original data was impossible given the processing power we had, due to the enormous size of the dataset."
      ],
      "id": "812768da"
    },
    {
      "cell_type": "code",
      "source": [
        "#Started of with fitment of the smaller dataset to observe which model performs better."
      ],
      "metadata": {
        "id": "kv032tIkSd3r"
      },
      "id": "kv032tIkSd3r",
      "execution_count": null,
      "outputs": []
    },
    {
      "cell_type": "code",
      "execution_count": null,
      "metadata": {
        "id": "16499621"
      },
      "outputs": [],
      "source": [
        "model.fit(SmallSample)"
      ],
      "id": "16499621"
    },
    {
      "cell_type": "code",
      "execution_count": null,
      "metadata": {
        "colab": {
          "background_save": true
        },
        "id": "42ea453e"
      },
      "outputs": [],
      "source": [
        "model1.fit(SmallSample)"
      ],
      "id": "42ea453e"
    },
    {
      "cell_type": "code",
      "execution_count": null,
      "metadata": {
        "id": "ea14dc76"
      },
      "outputs": [],
      "source": [
        "model2.fit(SmallSample)"
      ],
      "id": "ea14dc76"
    },
    {
      "cell_type": "code",
      "execution_count": null,
      "metadata": {
        "id": "1287ad2a"
      },
      "outputs": [],
      "source": [
        "model3.fit(SmallSample)"
      ],
      "id": "1287ad2a"
    },
    {
      "cell_type": "code",
      "execution_count": null,
      "metadata": {
        "id": "9154120c"
      },
      "outputs": [],
      "source": [
        "model4.fit(SmallSample)"
      ],
      "id": "9154120c"
    },
    {
      "cell_type": "code",
      "execution_count": null,
      "metadata": {
        "id": "32bc3b10"
      },
      "outputs": [],
      "source": [
        "model5.fit(SmallSample)"
      ],
      "id": "32bc3b10"
    },
    {
      "cell_type": "code",
      "execution_count": null,
      "metadata": {
        "id": "35888696"
      },
      "outputs": [],
      "source": [
        "model6.fit(SmallSample)"
      ],
      "id": "35888696"
    },
    {
      "cell_type": "code",
      "execution_count": null,
      "metadata": {
        "id": "71821d14"
      },
      "outputs": [],
      "source": [
        "# Sample Synthetic Data generated"
      ],
      "id": "71821d14"
    },
    {
      "cell_type": "code",
      "execution_count": null,
      "metadata": {
        "id": "c1886ba0"
      },
      "outputs": [],
      "source": [
        "new_data = model.sample(num_rows=1000)"
      ],
      "id": "c1886ba0"
    },
    {
      "cell_type": "code",
      "execution_count": null,
      "metadata": {
        "id": "8e55d19d"
      },
      "outputs": [],
      "source": [
        "new_data1 = model1.sample(num_rows=1000)"
      ],
      "id": "8e55d19d"
    },
    {
      "cell_type": "code",
      "execution_count": null,
      "metadata": {
        "id": "426cf86a"
      },
      "outputs": [],
      "source": [
        "new_data2 = model2.sample(num_rows=1000)"
      ],
      "id": "426cf86a"
    },
    {
      "cell_type": "code",
      "execution_count": null,
      "metadata": {
        "id": "1702f112"
      },
      "outputs": [],
      "source": [
        "new_data3 = model3.sample(num_rows=1000)"
      ],
      "id": "1702f112"
    },
    {
      "cell_type": "code",
      "execution_count": null,
      "metadata": {
        "id": "0893e95d"
      },
      "outputs": [],
      "source": [
        "new_data4 = model4.sample(num_rows=1000)"
      ],
      "id": "0893e95d"
    },
    {
      "cell_type": "code",
      "execution_count": null,
      "metadata": {
        "id": "2be35f92"
      },
      "outputs": [],
      "source": [
        "new_data5 = model5.sample(num_rows=1000)"
      ],
      "id": "2be35f92"
    },
    {
      "cell_type": "code",
      "execution_count": null,
      "metadata": {
        "id": "5f87a088"
      },
      "outputs": [],
      "source": [
        "new_data6 = model6.sample(num_rows=1000)"
      ],
      "id": "5f87a088"
    },
    {
      "cell_type": "code",
      "execution_count": null,
      "metadata": {
        "id": "8bd60f4f"
      },
      "outputs": [],
      "source": [
        "#Inspect Synthetic Data"
      ],
      "id": "8bd60f4f"
    },
    {
      "cell_type": "code",
      "execution_count": null,
      "metadata": {
        "id": "7eafe220"
      },
      "outputs": [],
      "source": [
        "SampleLoans.head()"
      ],
      "id": "7eafe220"
    },
    {
      "cell_type": "code",
      "execution_count": null,
      "metadata": {
        "id": "d98f4f0f"
      },
      "outputs": [],
      "source": [
        "new_data.head()"
      ],
      "id": "d98f4f0f"
    },
    {
      "cell_type": "code",
      "execution_count": null,
      "metadata": {
        "id": "c9f6d189"
      },
      "outputs": [],
      "source": [
        "new_data1.head()"
      ],
      "id": "c9f6d189"
    },
    {
      "cell_type": "code",
      "execution_count": null,
      "metadata": {
        "id": "b97e18b2"
      },
      "outputs": [],
      "source": [
        "new_data2.head()"
      ],
      "id": "b97e18b2"
    },
    {
      "cell_type": "code",
      "execution_count": null,
      "metadata": {
        "id": "4b1c7502"
      },
      "outputs": [],
      "source": [
        "new_data3.head()"
      ],
      "id": "4b1c7502"
    },
    {
      "cell_type": "code",
      "execution_count": null,
      "metadata": {
        "id": "7a2ddea7"
      },
      "outputs": [],
      "source": [
        "new_data4.head()"
      ],
      "id": "7a2ddea7"
    },
    {
      "cell_type": "code",
      "execution_count": null,
      "metadata": {
        "id": "31a4d3eb"
      },
      "outputs": [],
      "source": [
        "new_data5.head()"
      ],
      "id": "31a4d3eb"
    },
    {
      "cell_type": "code",
      "execution_count": null,
      "metadata": {
        "id": "601d16ef"
      },
      "outputs": [],
      "source": [
        "new_data6.head()"
      ],
      "id": "601d16ef"
    },
    {
      "cell_type": "code",
      "execution_count": null,
      "metadata": {
        "id": "112a85d7"
      },
      "outputs": [],
      "source": [
        "# Evaluate Synthetic Data"
      ],
      "id": "112a85d7"
    },
    {
      "cell_type": "code",
      "execution_count": null,
      "metadata": {
        "id": "f0daf0a0"
      },
      "outputs": [],
      "source": [
        "evaluate(new_data, SampleLoans)"
      ],
      "id": "f0daf0a0"
    },
    {
      "cell_type": "code",
      "execution_count": null,
      "metadata": {
        "id": "20a59305"
      },
      "outputs": [],
      "source": [
        "evaluate(new_data1, SampleLoans)"
      ],
      "id": "20a59305"
    },
    {
      "cell_type": "code",
      "execution_count": null,
      "metadata": {
        "id": "685d5e28"
      },
      "outputs": [],
      "source": [
        "evaluate(new_data2, SampleLoans)"
      ],
      "id": "685d5e28"
    },
    {
      "cell_type": "code",
      "execution_count": null,
      "metadata": {
        "id": "e57535bb"
      },
      "outputs": [],
      "source": [
        "evaluate(new_data3, SampleLoans)"
      ],
      "id": "e57535bb"
    },
    {
      "cell_type": "code",
      "execution_count": null,
      "metadata": {
        "id": "31558abe"
      },
      "outputs": [],
      "source": [
        "evaluate(new_data4, SampleLoans)"
      ],
      "id": "31558abe"
    },
    {
      "cell_type": "code",
      "execution_count": null,
      "metadata": {
        "id": "de18a895"
      },
      "outputs": [],
      "source": [
        "evaluate(new_data5, SampleLoans)"
      ],
      "id": "de18a895"
    },
    {
      "cell_type": "code",
      "execution_count": null,
      "metadata": {
        "id": "cb3103bd"
      },
      "outputs": [],
      "source": [
        "evaluate(new_data6, SampleLoans)"
      ],
      "id": "cb3103bd"
    },
    {
      "cell_type": "code",
      "execution_count": null,
      "metadata": {
        "id": "3be37207"
      },
      "outputs": [],
      "source": [
        "#Observe Correlations:"
      ],
      "id": "3be37207"
    },
    {
      "cell_type": "code",
      "execution_count": null,
      "metadata": {
        "id": "7d96df71"
      },
      "outputs": [],
      "source": [
        "print(SampleLoans.corr())"
      ],
      "id": "7d96df71"
    },
    {
      "cell_type": "code",
      "execution_count": null,
      "metadata": {
        "id": "41932215"
      },
      "outputs": [],
      "source": [
        "print(new_data.corr())"
      ],
      "id": "41932215"
    },
    {
      "cell_type": "code",
      "execution_count": null,
      "metadata": {
        "id": "f7f51db5"
      },
      "outputs": [],
      "source": [
        "print(new_data1.corr())"
      ],
      "id": "f7f51db5"
    },
    {
      "cell_type": "code",
      "execution_count": null,
      "metadata": {
        "id": "f597a1f9"
      },
      "outputs": [],
      "source": [
        "print(new_data2.corr())"
      ],
      "id": "f597a1f9"
    },
    {
      "cell_type": "code",
      "execution_count": null,
      "metadata": {
        "id": "d2c67050"
      },
      "outputs": [],
      "source": [
        "print(new_data3.corr())"
      ],
      "id": "d2c67050"
    },
    {
      "cell_type": "code",
      "execution_count": null,
      "metadata": {
        "id": "0f3ede16"
      },
      "outputs": [],
      "source": [
        "print(new_data4.corr())"
      ],
      "id": "0f3ede16"
    },
    {
      "cell_type": "code",
      "execution_count": null,
      "metadata": {
        "id": "f78b336b"
      },
      "outputs": [],
      "source": [
        "print(new_data5.corr())"
      ],
      "id": "f78b336b"
    },
    {
      "cell_type": "code",
      "execution_count": null,
      "metadata": {
        "id": "905bb22c"
      },
      "outputs": [],
      "source": [
        "print(new_data6.corr())"
      ],
      "id": "905bb22c"
    },
    {
      "cell_type": "code",
      "source": [
        "#It was evident that Model 3 (i.e. model1) performs better so we proceeded on with fitting the slightly bigger sample 'Loans Sample' to model1."
      ],
      "metadata": {
        "id": "FL0Kb4mIS8vk"
      },
      "id": "FL0Kb4mIS8vk",
      "execution_count": null,
      "outputs": []
    },
    {
      "cell_type": "code",
      "source": [
        "model1b = CTGAN(constraints=constraints2)"
      ],
      "metadata": {
        "id": "NFjrl4YZTus6"
      },
      "id": "NFjrl4YZTus6",
      "execution_count": null,
      "outputs": []
    },
    {
      "cell_type": "code",
      "source": [
        "model1b.fit(SampleLoans)"
      ],
      "metadata": {
        "id": "VcmztzIQTLbV"
      },
      "id": "VcmztzIQTLbV",
      "execution_count": null,
      "outputs": []
    },
    {
      "cell_type": "code",
      "source": [
        "#Due to the increased size of the dataset we sample a slightly larger sample from it. 4000rows>1000rows."
      ],
      "metadata": {
        "id": "IKbcrn13TOOx"
      },
      "id": "IKbcrn13TOOx",
      "execution_count": null,
      "outputs": []
    },
    {
      "cell_type": "code",
      "source": [
        "new_data1b = model1b.sample(num_rows=1000)"
      ],
      "metadata": {
        "id": "kjaNDLk0RdRI"
      },
      "id": "kjaNDLk0RdRI",
      "execution_count": null,
      "outputs": []
    },
    {
      "cell_type": "code",
      "source": [
        "new_data1bb = model1b.sample(num_rows=4000)"
      ],
      "metadata": {
        "id": "1hP7Ha_rTY2Q"
      },
      "id": "1hP7Ha_rTY2Q",
      "execution_count": null,
      "outputs": []
    },
    {
      "cell_type": "code",
      "source": [
        "#Evaluation of results."
      ],
      "metadata": {
        "id": "sRLdYio0ZcT9"
      },
      "id": "sRLdYio0ZcT9",
      "execution_count": null,
      "outputs": []
    },
    {
      "cell_type": "code",
      "source": [
        "evaluate(newdata1b,SampleLoans)"
      ],
      "metadata": {
        "id": "lWVGDjPQRfGq"
      },
      "id": "lWVGDjPQRfGq",
      "execution_count": null,
      "outputs": []
    },
    {
      "cell_type": "code",
      "source": [
        "evaluate(newdata1bb,SampleLoans)"
      ],
      "metadata": {
        "id": "UXSURwXhZbeQ"
      },
      "id": "UXSURwXhZbeQ",
      "execution_count": null,
      "outputs": []
    },
    {
      "cell_type": "code",
      "source": [
        "#Visual and Statistical Evaluation"
      ],
      "metadata": {
        "id": "cC7DVTgpqcnA"
      },
      "id": "cC7DVTgpqcnA",
      "execution_count": null,
      "outputs": []
    },
    {
      "cell_type": "code",
      "source": [
        "#Numerical variables statistics of original sample and synthetic data:"
      ],
      "metadata": {
        "id": "p02h_KBmaCrz"
      },
      "id": "p02h_KBmaCrz",
      "execution_count": null,
      "outputs": []
    },
    {
      "cell_type": "code",
      "source": [
        "SampleLoans[['loan_amnt','installment','int_rate','annual_inc','total_pymnt']].describe().transpose()"
      ],
      "metadata": {
        "id": "gf_Vp7YafVIH"
      },
      "id": "gf_Vp7YafVIH",
      "execution_count": null,
      "outputs": []
    },
    {
      "cell_type": "code",
      "source": [
        "new_data1bb[['loan_amnt','installment','int_rate','annual_inc','total_pymnt']].describe().transpose()"
      ],
      "metadata": {
        "id": "Z1vpm_8efU61"
      },
      "id": "Z1vpm_8efU61",
      "execution_count": null,
      "outputs": []
    },
    {
      "cell_type": "code",
      "source": [
        "#Categorical variables statistics of original sample and synthetic data:"
      ],
      "metadata": {
        "id": "8eFS2-D3aEb_"
      },
      "id": "8eFS2-D3aEb_",
      "execution_count": null,
      "outputs": []
    },
    {
      "cell_type": "code",
      "source": [
        "SampleLoans[['emp_length','verification_status','loan_status','addr_state','home_ownership']].describe(include=object).transpose()"
      ],
      "metadata": {
        "id": "gr-yvTv0aFOO"
      },
      "id": "gr-yvTv0aFOO",
      "execution_count": null,
      "outputs": []
    },
    {
      "cell_type": "code",
      "source": [
        "new_data1bb[['emp_length','verification_status','loan_status','addr_state','home_ownership']].describe(include=object).transpose()"
      ],
      "metadata": {
        "id": "zJj5ztKKaFu1"
      },
      "id": "zJj5ztKKaFu1",
      "execution_count": null,
      "outputs": []
    },
    {
      "cell_type": "code",
      "source": [
        "#Defining variables to include in comparative correlation matrix."
      ],
      "metadata": {
        "id": "uYGWZwcgaZrU"
      },
      "id": "uYGWZwcgaZrU",
      "execution_count": null,
      "outputs": []
    },
    {
      "cell_type": "code",
      "source": [
        "OrigCorrMatrix = SampleLoans[['loan_amnt','funded_amnt','int_rate','installment', 'annual_inc','total_pymnt']].copy()"
      ],
      "metadata": {
        "id": "O6AnTjv5hnKt"
      },
      "id": "O6AnTjv5hnKt",
      "execution_count": null,
      "outputs": []
    },
    {
      "cell_type": "code",
      "source": [
        "SynthCorrMatrix = new_data1bb[['loan_amnt','funded_amnt','int_rate','installment', 'annual_inc','total_pymnt']].copy()"
      ],
      "metadata": {
        "id": "iWm3vZJEiFjx"
      },
      "id": "iWm3vZJEiFjx",
      "execution_count": null,
      "outputs": []
    },
    {
      "cell_type": "code",
      "source": [
        "#Correlation matrix of chosen variables from synthetic and original distribution"
      ],
      "metadata": {
        "id": "3gYbkSiWagJM"
      },
      "id": "3gYbkSiWagJM",
      "execution_count": null,
      "outputs": []
    },
    {
      "cell_type": "code",
      "source": [
        "corrmatrix = OrigCorrMatrix.corr()\n",
        "plt.figure(figsize=(10, 8))\n",
        "sns.heatmap(corrmatrix, annot=True)\n",
        "plt.savefig('Original Sample Correlation Matrix.png')"
      ],
      "metadata": {
        "id": "u6Lsp5CeiQdY"
      },
      "id": "u6Lsp5CeiQdY",
      "execution_count": null,
      "outputs": []
    },
    {
      "cell_type": "code",
      "source": [
        "corrmatrix = SynthCorrMatrix.corr()\n",
        "plt.figure(figsize=(10, 8))\n",
        "sns.heatmap(corrmatrix, annot=True)\n",
        "plt.savefig('Synthetic Sample Correlation Matrix.png')"
      ],
      "metadata": {
        "id": "Fvt68BPkiQMu"
      },
      "id": "Fvt68BPkiQMu",
      "execution_count": null,
      "outputs": []
    },
    {
      "cell_type": "code",
      "source": [
        "#Employ Table Evaluator."
      ],
      "metadata": {
        "id": "KAvXCz66qx6z"
      },
      "id": "KAvXCz66qx6z",
      "execution_count": null,
      "outputs": []
    },
    {
      "cell_type": "code",
      "source": [
        "#Need to define categorical columns to feed in the table evaluator"
      ],
      "metadata": {
        "id": "SYIifmWNq04e"
      },
      "id": "SYIifmWNq04e",
      "execution_count": null,
      "outputs": []
    },
    {
      "cell_type": "code",
      "source": [
        "CatColumns=['term','sub_grade','emp_length','home_ownership','loan_status','verification_status','pymnt_plan','purpose','addr_state','initial_list_status','application_type','Last_Major_Derog_Length','Last_Delinq_Length','Last_Record_Length']"
      ],
      "metadata": {
        "id": "oq1dhYwAvut1"
      },
      "id": "oq1dhYwAvut1",
      "execution_count": null,
      "outputs": []
    },
    {
      "cell_type": "code",
      "source": [
        "table_evaluator = TableEvaluator(SampleLoans, new_data1bb, cat_cols=CatColumns)"
      ],
      "metadata": {
        "id": "CjLijt_Pvvyr"
      },
      "id": "CjLijt_Pvvyr",
      "execution_count": null,
      "outputs": []
    },
    {
      "cell_type": "code",
      "source": [
        "table_evaluator.visual_evaluation()"
      ],
      "metadata": {
        "id": "UabJ9Tpuv34k"
      },
      "id": "UabJ9Tpuv34k",
      "execution_count": null,
      "outputs": []
    },
    {
      "cell_type": "code",
      "source": [
        "#Test ability to classify Loan status"
      ],
      "metadata": {
        "id": "ElrCXSQTv61e"
      },
      "id": "ElrCXSQTv61e",
      "execution_count": null,
      "outputs": []
    },
    {
      "cell_type": "code",
      "source": [
        "table_evaluator.evaluate(target_col='loan_status')"
      ],
      "metadata": {
        "id": "3kDEWP3Sv-iQ"
      },
      "id": "3kDEWP3Sv-iQ",
      "execution_count": null,
      "outputs": []
    }
  ],
  "metadata": {
    "colab": {
      "name": "MastersProject.ipynb",
      "provenance": [],
      "collapsed_sections": []
    },
    "kernelspec": {
      "display_name": "Python 3",
      "language": "python",
      "name": "python3"
    },
    "language_info": {
      "codemirror_mode": {
        "name": "ipython",
        "version": 3
      },
      "file_extension": ".py",
      "mimetype": "text/x-python",
      "name": "python",
      "nbconvert_exporter": "python",
      "pygments_lexer": "ipython3",
      "version": "3.8.8"
    }
  },
  "nbformat": 4,
  "nbformat_minor": 5
}
