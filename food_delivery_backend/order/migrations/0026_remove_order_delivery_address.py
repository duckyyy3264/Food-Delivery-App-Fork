# Generated by Django 5.1 on 2024-08-25 12:15

from django.db import migrations


class Migration(migrations.Migration):
    dependencies = [
        ("order", "0025_alter_order_delivery_fee"),
    ]

    operations = [
        migrations.RemoveField(
            model_name="order",
            name="delivery_address",
        ),
    ]
