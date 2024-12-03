import json
from datetime import datetime

from django.db import models


class Place(models.Model):
    ID = models.IntegerField(primary_key=True)
    Name = models.CharField(max_length=255)
    Category = models.CharField(max_length=100)
    Description = models.TextField()
    OpeningTime = models.CharField(max_length=500)
    Website = models.URLField(max_length=200, null=True, blank=True)
    PhoneNumber = models.CharField(max_length=20, null=True, blank=True)
    Location = models.CharField(max_length=100)
    Address = models.CharField(max_length=255)
    Toilet = models.BooleanField(default=False)
    Accessibility = models.BooleanField(default=False)
    Animals = models.BooleanField(default=False)
    Image = models.CharField(max_length=250)
    Icon = models.CharField(max_length=250)
    VisitTime = models.CharField(max_length=100)
    Map_Priority = models.IntegerField(default=0)
    place_id = models.CharField(max_length=200, blank=True, null=True)
    average_rating = models.FloatField(null=True, blank=True)
    user_rating_accessibility = models.FloatField(null=True, blank=True)
    num_positive_comments = models.IntegerField(null=True, default=0)
    num_negative_comments = models.IntegerField(null=True,default=0)
    wheelchair_accessible_entrance = models.BooleanField(null=True, blank=True)

    def __str__(self):
        return self.Name

from django.db import models
#
#
# class Place(models.Model):
#     ID = models.AutoField(primary_key=True)
#     Name = models.CharField(max_length=255)
#     Category = models.CharField(max_length=100, choices=[
#         ('Historical Sites', 'Historical Sites'),
#         ('Museums', 'Museums'),
#         ('Parks', 'Parks'),
#         ('Churches', 'Churches'),
#         ('Markets', 'Markets'),
#         ('Scenic Spots', 'Scenic Spots'),
#         ('Beaches', 'Beaches'),
#     ])
#     Description = models.TextField()
#     OpeningTime = models.JSONField()  # Storing opening times as JSON for flexibility
#     Website = models.URLField(max_length=200, null=True, blank=True)
#     PhoneNumber = models.CharField(max_length=20, null=True, blank=True)
#     Location = models.CharField(max_length=100)
#     Address = models.CharField(max_length=255)
#     Toilet = models.BooleanField(default=False)
#     Accessibility = models.BooleanField(default=False)
#     Animals = models.BooleanField(default=False)
#     Image = models.CharField(max_length=250)
#     Icon = models.CharField(max_length=250)
#     VisitTime = models.IntegerField()  # Recommended visit duration in minutes
#     Map_Priority = models.IntegerField(default=0)
#     place_id = models.CharField(max_length=200, blank=True, null=True)
#     average_rating = models.FloatField(null=True, blank=True)
#     user_rating_accessibility = models.FloatField(null=True, blank=True)
#     num_positive_comments = models.IntegerField(default=0)
#     num_negative_comments = models.IntegerField(default=0)
#     wheelchair_accessible_entrance = models.BooleanField(null=True, blank=True)
#
#     def __str__(self):
#         return f"{self.Name} ({self.Category})"
#
#     @property
#     def positive_comments_ratio(self):
#         """Calculate the ratio of positive comments."""
#         total_comments = self.num_positive_comments + self.num_negative_comments
#         return (self.num_positive_comments / total_comments) if total_comments > 0 else 0
#
#     @property
#     def score(self, alpha=0.4, beta=0.4, gamma=0.2):
#         """
#         Calculate a composite score for the POI based on:
#         - average_rating
#         - positive_comments_ratio
#         - accessibility
#         """
#         return (
#                 alpha * (self.average_rating or 0) +
#                 beta * self.positive_comments_ratio +
#                 gamma * int(self.Accessibility)
#         )
#

class Locations(models.Model):
    name = models.CharField(max_length=500)
    category = models.CharField(max_length=500, blank=True, null=True)
    description = models.CharField(max_length=300, blank=True, null=True)
    OpeningTime = models.JSONField(blank=True, null=True)
    website_address = models.URLField(blank=True, null=True)
    phone_number = models.CharField(max_length=50, blank=True, null=True)
    zipcode = models.CharField(max_length=200, blank=True, null=True)
    city = models.CharField(max_length=200, blank=True, null=True)
    country = models.CharField(max_length=200, blank=True, null=True)
    address = models.CharField(max_length=200, blank=True, null=True)
    created_at = models.DateTimeField(auto_now_add=True, blank=True, null=True)
    edited_at = models.DateTimeField(auto_now=True)
    toilet = models.BooleanField(null=True)
    accessibility = models.BooleanField(null=True)
    animals = models.BooleanField(null=True)
    image = models.ImageField(upload_to='static/images/', null=True, blank=True)
    lat = models.CharField(max_length=200, blank=True, null=True)
    lng = models.CharField(max_length=200, blank=True, null=True)
    place_id = models.CharField(max_length=200, blank=True, null=True)
    icon_image = models.CharField(max_length=200, blank=True, null=True)
    average_rating = models.FloatField(null=True, blank=True)
    user_rating_accessibility = models.FloatField(null=True, blank=True)
    num_positive_comments = models.IntegerField(default=0)
    num_negative_comments = models.IntegerField(default=0)
    wheelchair_accessible_entrance = models.BooleanField(null=True, blank=True)
    visitTime = models.CharField(max_length=200, blank=True, null=True)

    def __str__(self):
        return self.name


class Distances(models.Model):
    from_location = models.ForeignKey(Locations, related_name='from_location', on_delete=models.CASCADE)
    to_location = models.ForeignKey(Locations, related_name='to_location', on_delete=models.CASCADE)
    mode = models.CharField(max_length=200, blank=True, null=True)
    distance_km = models.DecimalField(max_digits=10, decimal_places=2)
    duration_mins = models.DecimalField(max_digits=10, decimal_places=2)
    duration_traffic_mins = models.DecimalField(max_digits=10, decimal_places=2, blank=True, null=True)
    created_at = models.DateTimeField(auto_now_add=True, blank=True, null=True)
    edited_at = models.DateTimeField(auto_now=True)


class TravelPreference(models.Model):
    departure_date = models.DateField()
    departure_time = models.CharField(max_length=5)  # Assuming the format HH:MM AM/PM, e.g., '10:30 AM'
    moving_preference = models.CharField(max_length=20, choices=[
        ('car', 'Car'),
        ('bus', 'Bus'),
        ('train', 'Train'),
        ('plane', 'Plane'),
        ('bike', 'Bike'),
        ('walking', 'Walking'),
        ('other', 'Other')
    ])
    main_interests = models.CharField(max_length=50, choices=[
        ('nature', 'Nature and Outdoor Activities'),
        ('culture', 'Cultural Experiences'),
        ('adventure', 'Adventure Sports'),
        ('food', 'Food and Culinary'),
        ('history', 'Historical Sites'),
        ('shopping', 'Shopping'),
        ('other', 'Other')
    ])

    def __str__(self):
        return f"Travel Preference #{self.id}"




class QValue(models.Model):
    age = models.IntegerField()
    nationality = models.CharField(max_length=50)
    interests = models.JSONField()  # Store interests as JSON
    action = models.JSONField()      # Store action as JSON
    q_value = models.FloatField(default=0)  # Q-value for this state-action pair

    class Meta:
        unique_together = ('age', 'nationality', 'interests', 'action')
