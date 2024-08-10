# Generated by Django 3.2.7 on 2024-08-07 03:38

import deliverer.models.deliverer
from django.db import migrations, models


class Migration(migrations.Migration):

    dependencies = [
        ('deliverer', '0005_alter_deliverer_rating_counts'),
    ]

    operations = [
        migrations.AlterField(
            model_name='deliverer',
            name='rating',
            field=models.DecimalField(blank=True, decimal_places=1, default=0, max_digits=3, null=True),
        ),
        migrations.AlterField(
            model_name='deliverer',
            name='rating_counts',
            field=models.JSONField(blank=True, default=deliverer.models.deliverer.default_rating_counts, null=True),
        ),
        migrations.AlterField(
            model_name='deliverer',
            name='total_reviews',
            field=models.IntegerField(blank=True, default=0, null=True),
        ),
    ]