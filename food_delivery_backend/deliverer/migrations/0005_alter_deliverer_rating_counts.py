# Generated by Django 3.2.7 on 2024-08-07 01:55

import deliverer.models.deliverer
from django.db import migrations, models


class Migration(migrations.Migration):

    dependencies = [
        ('deliverer', '0004_deliverer_rating_counts'),
    ]

    operations = [
        migrations.AlterField(
            model_name='deliverer',
            name='rating_counts',
            field=models.JSONField(default=deliverer.models.deliverer.default_rating_counts),
        ),
    ]