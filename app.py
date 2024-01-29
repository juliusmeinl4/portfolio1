from flask import Flask, render_template, request
import requests
import base64
import xmltodict
import pandas as pd
import json
import xml.etree.ElementTree as ET
from datetime import datetime
import subprocess
import shlex
import os
import sqlite3
import plotly.express as px
from datetime import datetime, timedelta
from dateutil.relativedelta import relativedelta
import plotly.graph_objects as go
import io
import csv
from flask import Response
from flask import make_response
from flask_sqlalchemy import SQLAlchemy
import plotly
from dash import Dash
import dash_table
import dash_html_components as html



app = Flask(__name__)


app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///' + os.path.join(app.instance_path, 'sales.db')
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False
db = SQLAlchemy(app)

with app.app_context():
    db.create_all()

class Sales(db.Model):
    __tablename__ = 'sales'

    sku = db.Column(db.Text, primary_key=True)
    qty = db.Column(db.Integer)
    site = db.Column(db.Text)
    cost = db.Column(db.Float)
    category = db.Column(db.Text)
    subcategory = db.Column(db.Text)
    brand = db.Column(db.Text)
    date = db.Column(db.Text)  
    Year = db.Column(db.Integer)
    Month = db.Column(db.Text)
    total = db.Column(db.Float)


class WholesaleSkuSales(db.Model):
    __tablename__ = 'wholesaleskusales'

    sku = db.Column(db.Text, primary_key=True)
    qty = db.Column(db.Integer)
    category = db.Column(db.Text)
    brand = db.Column(db.Text)
    date = db.Column(db.Text)  
    Year = db.Column(db.Integer)
    Month = db.Column(db.Text)

class Inventory(db.Model):
    __tablename__ = 'Inventory'

    sku = db.Column(db.Text, primary_key=True)
    qty = db.Column(db.Float)
    subcategory = db.Column(db.Text)
    color = db.Column(db.Text)
    brand = db.Column(db.Text)


def calculate_total_sum(query, conn, params):
    df = pd.read_sql_query(query, conn, params=params)
    if df.empty:
        return 0  
    else:
        return df.iloc[0]['total_sales']

def calculate_total_qty_sum(start_date, end_date, selected_brand):
    query = Sales.query.with_entities(db.func.sum(Sales.qty).label('total_qty_sum'))
    query = query.filter(Sales.date >= start_date, Sales.date <= end_date)

    if selected_brand and selected_brand.lower() != "all":
        query = query.filter(db.func.lower(Sales.brand) == selected_brand.lower())

    result = query.first()
    return result.total_qty_sum if result.total_qty_sum else 0



def fetch_aggregated_sku_sales_data(start_date, end_date, selected_brand):
    query = Sales.query.with_entities(Sales.sku, db.func.sum(Sales.total).label('total_sales'))
    query = query.filter(Sales.date >= start_date, Sales.date <= end_date)

    if selected_brand and selected_brand.lower() != "all":
        query = query.filter(db.func.lower(Sales.brand) == selected_brand.lower())

    query = query.group_by(Sales.sku).order_by(db.desc('total_sales'))
    engine = db.get_engine(app)  
    return pd.read_sql_query(query.statement, engine)




def fetch_aggregated_sku_sales_data(start_date, end_date, selected_brand):
    query = Sales.query.with_entities(Sales.sku, db.func.sum(Sales.total).label('total_sales'))
    query = query.filter(Sales.date >= start_date, Sales.date <= end_date)

    if selected_brand and selected_brand.lower() != "all":
        query = query.filter(db.func.lower(Sales.brand) == selected_brand.lower())

    query = query.group_by(Sales.sku).order_by(db.desc('total_sales'))

    
    results = query.all()

    
    return pd.DataFrame(results, columns=['sku', 'total_sales'])



def fetch_aggregated_site_sales_data(start_date, end_date, selected_brand):
    query = Sales.query.with_entities(Sales.site, db.func.sum(Sales.total).label('total_sales'))
    query = query.filter(Sales.date >= start_date, Sales.date <= end_date)

    if selected_brand and selected_brand.lower() != "all":
        query = query.filter(db.func.lower(Sales.brand) == selected_brand.lower())

    query = query.group_by(Sales.site).order_by(db.desc('total_sales'))

    results = query.all()
    return pd.DataFrame(results, columns=['site', 'total_sales'])



def fetch_aggregated_ytd_site_sales_data(selected_brand):
    start_date = datetime(datetime.now().year, 1, 1).strftime('%Y-%m-%d')
    end_date = datetime.now().strftime('%Y-%m-%d')

    query = Sales.query.with_entities(Sales.site, db.func.sum(Sales.total).label('total_sales'))
    query = query.filter(Sales.date >= start_date, Sales.date <= end_date)

    if selected_brand and selected_brand.lower() != "all":
        query = query.filter(db.func.lower(Sales.brand) == selected_brand.lower())

    query = query.group_by(Sales.site).order_by(db.desc('total_sales'))

    results = query.all()
    return pd.DataFrame(results, columns=['site', 'total_sales'])





def fetch_ytd_site_sales_data(year, brand=None):
    start_date = datetime(year, 1, 1).strftime('%Y-%m-%d')
    end_date = datetime(year, 12, 31).strftime('%Y-%m-%d')

    query = Sales.query.with_entities(Sales.site, Sales.date, db.func.sum(Sales.total).label('total_sales'))
    query = query.filter(Sales.date.between(start_date, end_date))

    if brand:
        query = query.filter(db.func.lower(Sales.brand) == brand.lower())

    query = query.group_by(Sales.site, Sales.date).order_by(Sales.date)
    return pd.read_sql_query(query.statement, db.session.bind)


def normalize_dates_for_comparison(df, base_year):
    df['date'] = pd.to_datetime(df['date'])
    df['normalized_date'] = df['date'].apply(lambda x: x.replace(year=base_year))
    return df



def fetch_ytd_sales_data_by_site(year, site, brand):
    start_date = datetime(year, 1, 1).strftime('%Y-%m-%d')
    end_date = datetime(year, 12, 31).strftime('%Y-%m-%d')

    query = Sales.query.with_entities(Sales.date, db.func.sum(Sales.total).label('total_sales'))
    query = query.filter(Sales.date.between(start_date, end_date), Sales.site == site)

    if brand:
        query = query.filter(db.func.lower(Sales.brand) == brand.lower())

    query = query.group_by(Sales.date).order_by(Sales.date)
    results = query.all()

    
    df = pd.DataFrame(results, columns=['date', 'total_sales'])
    df['date'] = pd.to_datetime(df['date'])
    return df




def get_brand_list():
    brands = Sales.query.with_entities(Sales.brand).distinct().all()
    return [brand[0] for brand in brands]

def get_site_list():
    sites = Sales.query.with_entities(Sales.site).distinct().all()
    return [site[0] for site in sites]



def generate_ytd_sales_chart(df):
    fig = px.line(df, x='date', y='total_sales')
    return fig.to_html(full_html=False)

def generate_combined_ytd_sales_chart(ytd_sales_data):
    fig = go.Figure()

    for year, df in ytd_sales_data.items():
        if not df.empty:
            df['display_date'] = df['date'].apply(lambda x: x.replace(year=2000))
            line_color = 'black' if year == 2024 else None  
            fig.add_trace(go.Scatter(x=df['display_date'], y=df['total_sales'],
                                     mode='lines', name=str(year),
                                     line=dict(color=line_color),
                                     hovertemplate='%{x} <br>Total Sales: %{y}<extra>Year: ' + str(year) + '</extra>'))

    fig.update_layout(
        xaxis_title='Date',
        yaxis_title='Total Sales',
        legend_title='Year',
        xaxis=dict(tickformat="%b %d")
    )

    return fig.to_html(full_html=False)



def get_latest_date():
    latest_date = db.session.query(db.func.max(Sales.date)).scalar()
    return latest_date


def get_week_number(date_str):
    date_obj = datetime.strptime(date_str, '%Y-%m-%d')
    return date_obj.isocalendar()[1]


def fetch_sales_data_for_week(year, week_number, site, brand):
    start_date, end_date = week_to_dates(year, week_number)
    query = Sales.query.with_entities(
        Sales.site, Sales.brand, 
        db.func.sum(Sales.qty).label('total_qty'),  
        db.func.sum(Sales.total).label('total_sales')  
    )
    query = query.filter(Sales.date.between(start_date, end_date), Sales.site == site)

    if brand:
        query = query.filter(db.func.lower(Sales.brand) == brand.lower())

    query = query.group_by(Sales.site, Sales.brand)
    results = query.all()

    
    return pd.DataFrame(results, columns=['site', 'brand', 'total_qty', 'total_sales'])



def week_to_dates(year, week_number):

    first_day_of_year = datetime(year, 1, 1)

    
    day_of_week = first_day_of_year.isoweekday()

    
    if day_of_week != 1:
        first_day_of_year += timedelta(days=(8 - day_of_week))

    
    start_date = first_day_of_year + timedelta(days=(week_number - 1) * 7)

    
    end_date = start_date + timedelta(days=6)

    return start_date.strftime('%Y-%m-%d'), end_date.strftime('%Y-%m-%d')



def fetch_sales_data_for_latest_date(latest_date):
    query = Sales.query.with_entities(Sales.site, db.func.sum(Sales.total).label('total_sales'))
    query = query.filter(Sales.date == latest_date).group_by(Sales.site)

    results = query.all()
    return pd.DataFrame(results, columns=['site', 'total_sales'])



def fetch_sales_data_for_same_period(year, current_month, current_day, site, brand):
    start_date = f"{year}-01-01"
    end_date = f"{year}-{current_month}-{current_day}"

    query = Sales.query.with_entities(db.func.sum(Sales.total).label('total_sales'))
    query = query.filter(Sales.date.between(start_date, end_date), Sales.site == site)

    if brand:
        query = query.filter(db.func.lower(Sales.brand) == brand.lower())

    return query.scalar() or 0



def fetch_aggregated_category_sales_data(start_date, end_date, selected_brand):
    query = Sales.query.with_entities(Sales.category, db.func.sum(Sales.total).label('total_sales'))
    query = query.filter(Sales.date.between(start_date, end_date))

    if selected_brand and selected_brand.lower() != "all":
        query = query.filter(db.func.lower(Sales.brand) == selected_brand.lower())

    query = query.group_by(Sales.category).order_by(db.desc('total_sales'))
    results = query.all()

    
    return pd.DataFrame(results, columns=['category', 'total_sales'])




def generate_category_sales_chart(df):
    fig = px.bar(df, x='category', y='total_sales')
    fig.update_layout(xaxis_title='Category', yaxis_title='Total Sales')
    return fig.to_html(full_html=False)

def generate_subcategory_sales_pie_chart(df):
    fig = px.pie(df, names='subcategory', values='total_sales')
    fig.update_traces(textposition='inside', textinfo='percent+label')
    return fig.to_html(full_html=False)


def fetch_aggregated_subcategory_sales_data(start_date, end_date, selected_brand):
    query = Sales.query.with_entities(Sales.subcategory, db.func.sum(Sales.total).label('total_sales'))
    query = query.filter(Sales.date.between(start_date, end_date))

    if selected_brand and selected_brand.lower() != "all":
        query = query.filter(db.func.lower(Sales.brand) == selected_brand.lower())

    query = query.group_by(Sales.subcategory).order_by(db.desc('total_sales'))
    results = query.all()

    
    return pd.DataFrame(results, columns=['subcategory', 'total_sales'])



def generate_subcategory_sales_chart(df):
    fig = px.bar(df, x='subcategory', y='total_sales')
    fig.update_layout(xaxis_title='Subcategory', yaxis_title='Total Sales')
    return fig.to_html(full_html=False)

def fetch_best_seller_wholesale_skus(start_date, end_date, selected_brand):
    query = WholesaleSkuSales.query.with_entities(WholesaleSkuSales.sku, db.func.sum(WholesaleSkuSales.qty).label('total_qty'))
    query = query.filter(WholesaleSkuSales.date.between(start_date, end_date))

    if selected_brand and selected_brand.lower() != "all":
        query = query.filter(db.func.lower(WholesaleSkuSales.brand) == selected_brand.lower())

    query = query.group_by(WholesaleSkuSales.sku).order_by(db.desc('total_qty'))
    results = query.all()

    
    return pd.DataFrame(results, columns=['sku', 'total_qty'])


def fetch_sales_by_category(start_date, end_date, selected_brand):
    query = WholesaleSkuSales.query.with_entities(WholesaleSkuSales.category, db.func.sum(WholesaleSkuSales.qty).label('total_qty'))
    query = query.filter(WholesaleSkuSales.date.between(start_date, end_date))

    if selected_brand and selected_brand.lower() != "all":
        query = query.filter(db.func.lower(WholesaleSkuSales.brand) == selected_brand.lower())

    query = query.group_by(WholesaleSkuSales.category)
    results = query.all()

    
    return pd.DataFrame(results, columns=['category', 'total_qty'])



def create_sales_by_category_chart(start_date, end_date, selected_brand):
    
    sales_data = fetch_sales_by_category(start_date, end_date, selected_brand)

    
    fig = px.bar(
        sales_data,
        x='category',
        y='total_qty',
        labels={'total_qty': 'Total Quantity', 'category': 'Category'}
    )

    
    fig.update_layout(
        xaxis_title='Category',
        yaxis_title='Total Quantity',
        plot_bgcolor='white',
        showlegend=False
    )

    
    chart_html = fig.to_html(full_html=False)
    
    return chart_html

def get_latest_wholesale_date():
    latest_date = db.session.query(db.func.max(WholesaleSkuSales.date).label('latest_date')).first()
    return latest_date.latest_date if latest_date else None


def fetch_latest_date_sales_by_category(latest_date, selected_brand):
    query = WholesaleSkuSales.query.with_entities(
        WholesaleSkuSales.category, 
        db.func.sum(WholesaleSkuSales.qty).label('total_qty')
    )
    query = query.filter(WholesaleSkuSales.date == latest_date)

    if selected_brand and selected_brand.lower() != "all":
        query = query.filter(db.func.lower(WholesaleSkuSales.brand) == selected_brand.lower())

    query = query.group_by(WholesaleSkuSales.category)
    results = query.all()

    
    return pd.DataFrame(results, columns=['category', 'total_qty'])




@app.route('/download-wholesale-skus')
def download_wholesale_skus():
    start_date = request.args.get('start_date', '2022-01-01') 
    end_date = request.args.get('end_date', '2022-12-31') 
    selected_brand = request.args.get('brand', 'All') 

    data = fetch_best_seller_wholesale_skus(start_date, end_date, selected_brand)

    si = io.StringIO()
    cw = csv.writer(si)
    cw.writerow(['SKU', 'Total Quantity'])

    if not data.empty:
        for index, row in data.iterrows():
            cw.writerow([row['sku'], row['total_qty']])

    output = make_response(si.getvalue())
    output.headers["Content-Disposition"] = "attachment; filename=best_seller_wholesale_skus.csv"
    output.headers["Content-type"] = "text/csv"
    return output

def calculate_ytd_sales_sum(selected_brand):
    start_date = datetime(datetime.now().year, 1, 1)
    end_date = datetime.now()

    query = Sales.query.with_entities(db.func.sum(Sales.total).label('ytd_sales_sum'))
    query = query.filter(Sales.date.between(start_date, end_date))

    if selected_brand and selected_brand.lower() != "all":
        query = query.filter(db.func.lower(Sales.brand) == selected_brand.lower())

    result = query.first()
    return result.ytd_sales_sum if result.ytd_sales_sum is not None else 0



def calculate_last_year_sales_sum(selected_brand, start_date, end_date):
    start_date_obj = datetime.strptime(start_date, '%Y-%m-%d')
    end_date_obj = datetime.strptime(end_date, '%Y-%m-%d')
    last_year_start = start_date_obj.replace(year=start_date_obj.year - 1)
    last_year_end = end_date_obj.replace(year=end_date_obj.year - 1)

    query = Sales.query.with_entities(db.func.sum(Sales.total).label('last_year_sales_sum'))
    query = query.filter(Sales.date.between(last_year_start, last_year_end))

    if selected_brand and selected_brand.lower() != "all":
        query = query.filter(db.func.lower(Sales.brand) == selected_brand.lower())

    result = query.first()
    return result.last_year_sales_sum if result.last_year_sales_sum is not None else 0



def calculate_previous_year_sales_sum(selected_brand, start_date, end_date):
    start_date_obj = datetime.strptime(start_date, '%Y-%m-%d')
    end_date_obj = datetime.strptime(end_date, '%Y-%m-%d')
    previous_year_start = start_date_obj.replace(year=start_date_obj.year - 2)
    previous_year_end = end_date_obj.replace(year=end_date_obj.year - 2)

    query = Sales.query.with_entities(db.func.sum(Sales.total).label('previous_year_sales_sum'))
    query = query.filter(Sales.date.between(previous_year_start, previous_year_end))

    if selected_brand and selected_brand.lower() != "all":
        query = query.filter(db.func.lower(Sales.brand) == selected_brand.lower())

    result = query.first()
    return result.previous_year_sales_sum if result.previous_year_sales_sum is not None else 0



def calculate_percentage_difference(current_value, previous_value):
    if previous_value > 0:
        return ((current_value - previous_value) / previous_value) * 100
    return None

def fetch_yearly_sales_data(year, brand):
    sales_data = Sales.query.filter(
        db.extract('year', Sales.date) == year,
        db.func.lower(Sales.brand) == brand.lower() if brand and brand.lower() != "all" else True
    ).with_entities(
        Sales.date,
        db.func.sum(Sales.total).label('total_sales')
    ).group_by(Sales.date).all()

    
    if not sales_data:
        return pd.DataFrame(columns=['date', 'total_sales'])  

    df = pd.DataFrame(sales_data, columns=['date', 'total_sales'])

    
    if 'date' not in df.columns:
        raise ValueError("Date column not found in the DataFrame.")

    df['date'] = pd.to_datetime(df['date'])
    return df


def create_yearly_sales_charts(yearly_sales_data):
    fig = px.line()
    fig2 = px.line()
    for year, data in yearly_sales_data.items():
        normalized_date = data['date'].apply(lambda x: x.replace(year=2024)) if year != 2024 else data['date']
        fig.add_scatter(x=normalized_date, y=data['total_sales'], mode='lines', name=str(year))
        fig2.add_scatter(x=data['date'], y=data['total_sales'], mode='lines', name=str(year))
    
    fig.update_layout(xaxis_title='Date', yaxis_title='Total Sales', legend_title='Year')
    fig2.update_layout(xaxis_title='Date', yaxis_title='Total Sales', legend_title='Year')
    return fig, fig2


@app.route('/', methods=['GET', 'POST'])
def index():
    
    today = datetime.now().date()
    start_of_week = today - timedelta(days=today.weekday())  
    end_of_week = start_of_week + timedelta(days=6)

    default_start_date = start_of_week.strftime('%Y-%m-%d')
    default_end_date = end_of_week.strftime('%Y-%m-%d')

    start_date = request.args.get('start_date', default_start_date)
    end_date = request.args.get('end_date', default_end_date)
    selected_brand = request.args.get('brand', 'all').lower()

    
    if not selected_brand:
        selected_brand = 'all'

    
    prev_year_start = (datetime.strptime(start_date, '%Y-%m-%d') - relativedelta(years=1)).strftime('%Y-%m-%d')
    prev_year_end = (datetime.strptime(end_date, '%Y-%m-%d') - relativedelta(years=1)).strftime('%Y-%m-%d')
    two_years_ago_start = (datetime.strptime(start_date, '%Y-%m-%d') - relativedelta(years=2)).strftime('%Y-%m-%d')
    two_years_ago_end = (datetime.strptime(end_date, '%Y-%m-%d') - relativedelta(years=2)).strftime('%Y-%m-%d')

    
    query = Sales.query.with_entities(Sales.date.label('formatted_date'), db.func.sum(Sales.total).label('total_sales'))
    query = query.filter(Sales.date >= start_date, Sales.date <= end_date)

    if selected_brand != "all":
        query = query.filter(db.func.lower(Sales.brand) == selected_brand)

    line_chart_data = query.group_by(Sales.date).order_by(Sales.date).all()
    df = pd.DataFrame(line_chart_data)

    total_sales_sum = df['total_sales'].sum() if not df.empty else 0
    formatted_total_sales_sum = "{:,.2f}".format(total_sales_sum)






    
    if prev_year_start and prev_year_end:
        total_sales_sum_prev_year = Sales.query.filter(
            Sales.date.between(prev_year_start, prev_year_end),
            db.func.lower(Sales.brand) == selected_brand if selected_brand != "all" else True
        ).with_entities(db.func.sum(Sales.total)).scalar() or 0

        formatted_total_sales_sum_prev_year = "{:,.2f}".format(total_sales_sum_prev_year)

        total_sales_sum_two_years_ago = Sales.query.filter(
            Sales.date.between(two_years_ago_start, two_years_ago_end),
            db.func.lower(Sales.brand) == selected_brand if selected_brand != "all" else True
        ).with_entities(db.func.sum(Sales.total)).scalar() or 0

        formatted_total_sales_sum_two_years_ago = "{:,.2f}".format(total_sales_sum_two_years_ago)

        percentage_diff = calculate_percentage_difference(total_sales_sum, total_sales_sum_prev_year)
        formatted_percentage_diff = "{:.2f}".format(percentage_diff) if percentage_diff is not None else "N/A"

        percentage_diff_two_years = calculate_percentage_difference(total_sales_sum, total_sales_sum_two_years_ago)
        formatted_percentage_diff_two_years = "{:.2f}".format(percentage_diff_two_years) if percentage_diff_two_years is not None else "N/A"

        total_qty_sum_prev_year = calculate_total_qty_sum(prev_year_start, prev_year_end, selected_brand)
        total_qty_sum_two_years_ago = calculate_total_qty_sum(two_years_ago_start, two_years_ago_end, selected_brand)
    else:
        formatted_total_sales_sum_prev_year = "N/A"
        formatted_total_sales_sum_two_years_ago = "N/A"
        formatted_percentage_diff = "N/A"
        formatted_percentage_diff_two_years = "N/A"
        total_qty_sum_prev_year = 0
        total_qty_sum_two_years_ago = 0


    best_seller_site_result = Sales.query.filter(
        Sales.date.between(start_date, end_date),
        db.func.lower(Sales.brand) == selected_brand if selected_brand != "all" else True
    ).with_entities(
        Sales.site,
        db.func.sum(Sales.total).label('total_sales')
    ).group_by(Sales.site).order_by(db.desc('total_sales')).limit(1).first()

    if best_seller_site_result and best_seller_site_result[1] is not None:
        best_seller_site = best_seller_site_result[0]
        best_seller_site_sales = "{:,.2f}".format(best_seller_site_result[1])
    else:
        best_seller_site = "N/A"
        best_seller_site_sales = "N/A"


    
    best_seller_sku_result = Sales.query.filter(
        Sales.date.between(start_date, end_date),
        db.func.lower(Sales.brand) == selected_brand if selected_brand != "all" else True
    ).with_entities(
        Sales.sku,
        db.func.sum(Sales.qty).label('total_qty_sold')
    ).group_by(Sales.sku).order_by(db.desc('total_qty_sold')).limit(1).first()

    best_seller_sku, best_seller_qty = best_seller_sku_result if best_seller_sku_result else ("N/A", "N/A")



    total_qty_sum = calculate_total_qty_sum(start_date, end_date, selected_brand)





    
    percentage_qty_diff_prev_year = calculate_percentage_difference(total_qty_sum, total_qty_sum_prev_year)
    formatted_percentage_qty_diff_prev_year = "{:.2f}".format(percentage_qty_diff_prev_year) if percentage_qty_diff_prev_year is not None else "N/A"

    
    percentage_qty_diff_two_years = calculate_percentage_difference(total_qty_sum, total_qty_sum_two_years_ago)
    formatted_percentage_qty_diff_two_years = "{:.2f}".format(percentage_qty_diff_two_years) if percentage_qty_diff_two_years is not None else "N/A"

    
    yearly_sales_data = {}
    for year in [2021, 2022, 2023, 2024]:
        yearly_sales_data[year] = fetch_yearly_sales_data(year, selected_brand)

    fig, fig2 = create_yearly_sales_charts(yearly_sales_data)

    graph_html = fig.to_html(full_html=False)
    graph_html2 = fig2.to_html(full_html=False)



    
    
    aggregated_sku_sales_data = fetch_aggregated_sku_sales_data(start_date, end_date, selected_brand)

    
    sku_fig = px.bar(aggregated_sku_sales_data, x='sku', y='total_sales', color='sku')
    sku_fig.update_layout(xaxis_title='SKU', yaxis_title='Total Sales')
    sku_graph_html = sku_fig.to_html(full_html=False)

    
    


    
    aggregated_site_sales_data = fetch_aggregated_site_sales_data(start_date, end_date, selected_brand)

    
    site_fig = px.pie(aggregated_site_sales_data, values='total_sales', names='site')
    site_fig.update_traces(textposition='inside', textinfo='percent+label')
    site_graph_html = site_fig.to_html(full_html=False)

    
        
    ytd_sales_data = fetch_aggregated_ytd_site_sales_data(selected_brand)

    
    ytd_site_fig = px.pie(ytd_sales_data, values='total_sales', names='site')
    ytd_site_fig.update_traces(textposition='inside', textinfo='percent+label')
    ytd_site_graph_html = ytd_site_fig.to_html(full_html=False)


    
    latest_date = get_latest_date()
    latest_sales_data = fetch_sales_data_for_latest_date(latest_date)

    latest_sales_fig = px.bar(latest_sales_data, x='site', y='total_sales')
    latest_sales_graph_html = latest_sales_fig.to_html(full_html=False)


    
    site_list = get_site_list()
    brand_list = get_brand_list()
    selected_site = "source 1"  
    selected_ytd_brand = brand_list[0]  

    if request.method == 'POST':
        selected_site = request.form.get('site_dropdown', "source 1")
        selected_ytd_brand = request.form.get('ytd_brand_dropdown')
    elif request.method == 'GET':
        selected_site = request.args.get('site_dropdown', "source 1")

    current_year = datetime.now().year
    ytd_sales_data = {}

    
    for year in range(2021, current_year + 1):
        ytd_sales_data[year] = fetch_ytd_sales_data_by_site(year, selected_site, selected_ytd_brand)

    ytd_sales_chart = generate_combined_ytd_sales_chart(ytd_sales_data)


    current_year = datetime.now().year
    current_month = datetime.now().strftime('%m')
    current_day = datetime.now().strftime('%d')

    
    latest_date = get_latest_date()
    latest_sales_data = fetch_sales_data_for_latest_date(latest_date)


    
    same_period_sales_data = {}
    for year in range(current_year - 3, current_year + 1):
        sales_amount = fetch_sales_data_for_same_period(year, current_month, current_day, selected_site, selected_ytd_brand)
        same_period_sales_data[year] = "{:,.2f}".format(sales_amount)  

    
    current_week_number = get_week_number(latest_date)
    print(f"Latest Date: {latest_date}, Week Number: {current_week_number}")
    weekly_sales_data = {}
    for year in range(current_year - 3, current_year + 1):
        weekly_data = fetch_sales_data_for_week(year, current_week_number, selected_site, selected_ytd_brand)
        total_weekly_sales = weekly_data['total_sales'].sum() if not weekly_data.empty else 0
        weekly_sales_data[year] = "{:,.2f}".format(total_weekly_sales)

    current_year_sales = fetch_sales_data_for_same_period(current_year, current_month, current_day, selected_site, selected_ytd_brand)
    current_week_sales = fetch_sales_data_for_week(current_year, current_week_number, selected_site, selected_ytd_brand)['total_sales'].sum()
    start_date, end_date = week_to_dates(2024, current_week_number)
    print(f"Start Date: {start_date}, End Date: {end_date}")
    
    for year in range(current_year - 3, current_year + 1):
        
        sales_amount = fetch_sales_data_for_same_period(year, current_month, current_day, selected_site, selected_ytd_brand)
        percent_diff = ((sales_amount - current_year_sales) / current_year_sales * 100) if current_year_sales != 0 else 0
        same_period_sales_data[year] = f"{sales_amount:,.2f} ({percent_diff:+.2f}%)"

        
        weekly_data = fetch_sales_data_for_week(year, current_week_number, selected_site, selected_ytd_brand)['total_sales'].sum()
        weekly_percent_diff = ((weekly_data - current_week_sales) / current_week_sales * 100) if current_week_sales != 0 else 0
        weekly_sales_data[year] = f"{weekly_data:,.2f} ({weekly_percent_diff:+.2f}%)"


    
    weekly_qty_data = {}
    
    
    current_week_qty = fetch_sales_data_for_week(current_year, current_week_number, selected_site, selected_ytd_brand)['total_qty'].sum()

    
    for year in range(current_year - 3, current_year + 1):
        weekly_data = fetch_sales_data_for_week(year, current_week_number, selected_site, selected_ytd_brand)
        total_weekly_qty = weekly_data['total_qty'].sum() if not weekly_data.empty else 0
        weekly_qty_percent_diff = ((total_weekly_qty - current_week_qty) / current_week_qty * 100) if current_week_qty != 0 else 0
        weekly_qty_data[year] = f"{total_weekly_qty:,.2f} ({weekly_qty_percent_diff:+.2f}%)"

    
    category_sales_data = fetch_aggregated_category_sales_data(start_date, end_date, selected_brand)
    category_sales_chart_html = generate_category_sales_chart(category_sales_data)

    subcategory_sales_data = fetch_aggregated_subcategory_sales_data(start_date, end_date, selected_brand)
    subcategory_sales_chart_html = generate_subcategory_sales_chart(subcategory_sales_data)

    
    subcategory_sales_data = fetch_aggregated_subcategory_sales_data(start_date, end_date, selected_brand)
    subcategory_sales_pie_chart_html = generate_subcategory_sales_pie_chart(subcategory_sales_data)

    best_seller_wholesale_skus_data = fetch_best_seller_wholesale_skus(start_date, end_date, selected_brand)
    
    fig = px.bar(best_seller_wholesale_skus_data, x='sku', y='total_qty',
                 labels={'sku': 'SKU', 'total_qty': 'Total Quantity'})
    bar_chart_html_wholesale = fig.to_html(full_html=False)
    
    
    sales_by_category_data = fetch_sales_by_category(start_date, end_date, selected_brand)

    
    pie_fig = px.pie(sales_by_category_data, names='category', values='total_qty',
                     labels={'total_qty': 'Total Quantity'})
    pie_chart_html_category_wholesale = pie_fig.to_html(full_html=False)


    
    latest_date = get_latest_wholesale_date()

    
    latest_sales_by_category_data = fetch_latest_date_sales_by_category(latest_date, selected_brand)

    
    latest_pie_fig = px.pie(latest_sales_by_category_data, names='category', values='total_qty',
                            labels={'total_qty': 'Total Quantity'})
    latest_pie_chart_html = latest_pie_fig.to_html(full_html=False)

    chart_html_pie_wholesale = create_sales_by_category_chart(start_date, end_date, selected_brand)

    
    selected_brand = request.args.get('brand', 'All')  

    
    ytd_sales_sum = calculate_ytd_sales_sum(selected_brand)
    ytd_sales_sum = "${:,.2f}".format(ytd_sales_sum)

    
    last_year_sales_sum = calculate_last_year_sales_sum(selected_brand, start_date, end_date)
    last_year_sales_sum = "${:,.2f}".format(last_year_sales_sum)


    
    previous_year_sales_sum = calculate_previous_year_sales_sum(selected_brand, start_date, end_date)
    previous_year_sales_sum = "${:,.2f}".format(previous_year_sales_sum)

    return render_template('index.html', 
                           graph_html=graph_html, graph_html2= graph_html2,
                           total_sales_sum=formatted_total_sales_sum, 
                           total_sales_sum_prev_year=formatted_total_sales_sum_prev_year,
                           percentage_diff=formatted_percentage_diff,
                           total_sales_sum_two_years_ago=formatted_total_sales_sum_two_years_ago,  
                           percentage_diff_two_years=formatted_percentage_diff_two_years,  
                           total_qty_sum=total_qty_sum, 
                           start_date=start_date, 
                           end_date=end_date, 
                           selected_brand=selected_brand,
                           total_qty_sum_prev_year=total_qty_sum_prev_year,
                           total_qty_sum_two_years_ago=total_qty_sum_two_years_ago,
                           percentage_qty_diff_prev_year=formatted_percentage_qty_diff_prev_year,
                           percentage_qty_diff_two_years=formatted_percentage_qty_diff_two_years,
                           best_seller_sku=best_seller_sku,
                           best_seller_qty=best_seller_qty,
                           best_seller_site=best_seller_site,
                           best_seller_site_sales=best_seller_site_sales, sku_graph_html=sku_graph_html, site_graph_html=site_graph_html,                            
                           site_list=site_list,
                           brand_list=brand_list,
                           selected_site=selected_site,
                           selected_ytd_brand=selected_ytd_brand,
                           ytd_sales_chart=ytd_sales_chart, ytd_site_graph_html=ytd_site_graph_html, latest_sales_graph_html=latest_sales_graph_html,
                           same_period_sales_data=same_period_sales_data,
                           weekly_sales_data=weekly_sales_data,
                            weekly_qty_data=weekly_qty_data,
                            category_sales_chart_html=category_sales_chart_html,
                            subcategory_sales_chart_html=subcategory_sales_chart_html,
                            subcategory_sales_pie_chart_html=subcategory_sales_pie_chart_html,
                           best_seller_wholesale_skus_data=best_seller_wholesale_skus_data.to_dict('records'),
                           bar_chart_html_wholesale=bar_chart_html_wholesale,
                           pie_chart_html_category_wholesale=pie_chart_html_category_wholesale, ytd_sales_sum=ytd_sales_sum, previous_year_sales_sum=previous_year_sales_sum,
                           latest_pie_chart_html=latest_pie_chart_html, chart_html_pie_wholesale=chart_html_pie_wholesale, last_year_sales_sum=last_year_sales_sum
                           )



dash_app = Dash(__name__, server=app, routes_pathname_prefix='/dash/')


dash_app.layout = html.Div([
    dash_table.DataTable(
        id='table',
        
    )
])




@app.route('/inventory')
def show_inventory():
    inventory_items = Inventory.query.all()
    print(inventory_items)

    
    filtered_items = [item for item in inventory_items if item is not None]
    
    df = pd.DataFrame([(item.sku, item.qty, item.subcategory, item.color, item.brand) 
                       for item in filtered_items], 
                      columns=['sku', 'Quantity', 'Subcategory', 'Color', 'Brand'])

    
    dash_app.layout = html.Div([
        dash_table.DataTable(
            columns=[{'name': i, 'id': i} for i in df.columns],
            data=df.to_dict('records'),
            filter_action='native',
            style_table={'height': 400},
            style_data={
                'width': '150px', 'minWidth': '150px', 'maxWidth': '150px',
                'overflow': 'hidden',
                'textOverflow': 'ellipsis',
            }
        )
    ])
    
    return render_template('inventory.html')





if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8080)
