# Generated by Django 5.1 on 2024-08-22 12:26

from django.db import migrations, models


class Migration(migrations.Migration):
    dependencies = [
        ("order", "0015_remove_restaurantcart_raw_fee"),
    ]

    operations = [
        migrations.AlterField(
            model_name="restaurantcartdish",
            name="quantity",
            field=models.IntegerField(default=0),
        ),
    ]
